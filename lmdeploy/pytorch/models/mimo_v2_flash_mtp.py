# Copyright (c) OpenMMLab. All rights reserved.
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_mtp import DeepseekMTPModel
from .mimo_v2_flash import MiMoV2Attention, MiMoV2MLP, _dequantize_blocked_fp8, _get_norm_eps
from .patch import add_prefix


def _mimo_swa_kernel_window(total_window_size: int) -> tuple[int, int]:
    """Translate MiMo's total causal window into kernel left/right bounds."""
    if not isinstance(total_window_size, int) or total_window_size <= 0:
        raise ValueError(f'MiMo MTP requires a positive total SWA window, got {total_window_size!r}.')
    # Kernel bounds are inclusive: left=127 plus the current causal token
    # gives MiMo's configured total window of 128 tokens.
    return total_window_size - 1, 0


class MiMoV2FlashMTPLayer(nn.Module):
    """One MiMo prediction-depth layer backed by paged SWA KV."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        norm_eps = _get_norm_eps(config)

        self.enorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('enorm', prefix),
        )
        self.hnorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('hnorm', prefix),
        )
        # Checkpoint eh_proj is BF16 even though the dense MLP is FP8.
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
            dp_disable_tp=True,
            prefix=add_prefix('eh_proj', prefix),
        )

        block_prefix = add_prefix('mtp_block', prefix)
        self.self_attn = MiMoV2Attention(
            config,
            is_swa=True,
            quantize_o_proj=False,
            dtype=dtype,
            device=device,
            prefix=add_prefix('self_attn', block_prefix),
        )
        # The target's state ring already stores exactly 128 entries.  This
        # draft path uses generic paged attention, whose inclusive bounds
        # would otherwise mean 128 history tokens plus the current token.
        total_window_size = getattr(config, 'sliding_window_size', getattr(config, 'sliding_window', None))
        self.self_attn.attn_fwd.impl.sliding_window = _mimo_swa_kernel_window(total_window_size)
        self.mlp = MiMoV2MLP(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('mlp', block_prefix),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', block_prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', block_prefix),
        )
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('final_layernorm', prefix),
        )

        self.rotary_emb = build_rotary_embedding(
            dim=config.swa_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.swa_rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_value: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Fuse target state and run one dense SWA decoder layer."""
        del input_ids
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        hidden_states = self.self_attn(
            hidden_states,
            rotary_pos_emb=(cos[0], sin[0]),
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class MiMoV2FlashMultiTokenPredictor(nn.Module):
    """Three checkpoint prediction depths sharing target embed/head."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers != 3:
            raise ValueError(f'MiMo-V2-Flash requires 3 MTP layers, got {self.num_mtp_layers}.')
        self.embed_tokens = None
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): MiMoV2FlashMTPLayer(
                    config,
                    layer_idx,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix(f'layers.{layer_idx}', prefix),
                )
                for layer_idx in range(self.num_mtp_layers)
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: Any = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run the prediction layer and cache selected by ``spec_step_idx``."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise RuntimeError('MiMo MTP input embedding has not been bound to the target model.')
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[str(current_step_idx)](
            input_ids,
            position_ids,
            previous_hidden_states,
            past_key_values[current_step_idx],
            inputs_embeds,
            attn_metadata=attn_metadata,
        )

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the final norm belonging to the active prediction depth."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)].final_layernorm(hidden_states)

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Bind the target model's shared token embedding."""
        self.embed_tokens = embed_tokens

    def get_input_embeddings(self):
        """Return the target-owned shared token embedding."""
        return self.embed_tokens


class MiMoV2FlashMTPModel(DeepseekMTPModel):
    """LMDeploy draft model for MiMo-V2-Flash MTP.

    ``DeepseekMTPModel`` is reused only for its generic target-hidden-state
    CUDA Graph buffers and generation-input plumbing. MiMo replaces the model
    body, checkpoint contract, logits normalization and proposer protocol.
    """

    packed_modules_mapping = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def support_spec_decode_without_fa3(self) -> bool:
        """Allow MiMo draft Graphs to use Triton paged SWA."""
        return True

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> bool:
        """Enable Graph for replay-safe MiMo draft steps.

        Runtime graph shapes are captured synchronously during warmup, so DP
        ranks never insert rank-local capture-time TP collectives.
        """
        return super().support_cuda_graph(
            input_ids,
            position_ids,
            past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def get_cudagraph_capture_cache(self, past_key_values, spec_step_idx: int = 0):
        """Return the active depth cache that Graph capture must preserve."""
        depth = spec_step_idx % self.model.num_mtp_layers
        return past_key_values[depth]

    def get_cudagraph_extra_key(
        self,
        spec_step_idx: int = 0,
        input_ids: torch.Tensor | None = None,
        attn_metadata: Any = None,
        **kwargs,
    ) -> tuple[int, int]:
        """Select a stable Graph for each MiMo prediction depth.

        ``spec_step_idx`` selects both a decoder module and its KV-cache
        entry.  As a Python scalar it cannot be changed by copying tensor
        inputs before graph replay, so it must participate in the graph key.

        Query length already participates in the generic Graph key. Keeping
        this key independent of rank-local KV length prevents one DP rank from
        inserting capture-time TP collectives while another rank replays.
        """
        del input_ids, attn_metadata, kwargs
        return (spec_step_idx % self.model.num_mtp_layers, -1)

    def get_cudagraph_warmup_specs(self, max_query_len: int) -> tuple[tuple[int, int], ...]:
        """Return every query-length and prediction-depth graph used at
        runtime."""
        return tuple(
            (query_len, depth)
            for query_len in range(1, max_query_len + 1)
            for depth in range(self.model.num_mtp_layers)
        )

    @staticmethod
    def prepare_cudagraph_warmup_inputs(inputs, max_history_len: int):
        """Capture multi-token draft graphs for the full session range.

        Triton's paged varlen attention specializes its KV-block loop during
        CUDA Graph capture.  Capturing with the largest valid history keeps
        replay correct when a request grows later in the session.
        """
        if inputs.max_q_seqlen <= 1:
            return inputs

        if max_history_len < 0:
            raise ValueError(f'max_history_len must be non-negative, got {max_history_len}.')
        inputs.history_lengths.fill_(max_history_len)
        inputs.max_kv_seqlen += max_history_len
        inputs.sum_kv_seqlen += max_history_len * inputs.seq_length.numel()
        positions = torch.arange(
            max_history_len,
            max_history_len + inputs.max_q_seqlen,
            dtype=inputs.target_position_ids.dtype,
            device=inputs.target_position_ids.device,
        )
        inputs.target_position_ids.copy_(positions.repeat(inputs.seq_length.numel()).unsqueeze(0))
        return inputs

    @staticmethod
    def select_weight_paths(model_path: str, default_paths: Iterable[str]) -> tuple[str, ...]:
        """Select the standalone MiMo MTP checkpoint instead of target
        shards."""
        del default_paths
        mtp_path = Path(model_path) / 'model_mtp.safetensors'
        if not mtp_path.is_file():
            raise FileNotFoundError(f'MiMo MTP checkpoint was not found: {mtp_path}')
        return (str(mtp_path),)

    _MTP_TENSOR_SUFFIXES = frozenset(
        {
            'enorm.weight',
            'hnorm.weight',
            'eh_proj.weight',
            'input_layernorm.weight',
            'pre_mlp_layernorm.weight',
            'final_layernorm.weight',
            'self_attn.q_proj.weight',
            'self_attn.q_proj.weight_scale_inv',
            'self_attn.k_proj.weight',
            'self_attn.k_proj.weight_scale_inv',
            'self_attn.v_proj.weight',
            'self_attn.v_proj.weight_scale_inv',
            'self_attn.o_proj.weight',
            'self_attn.attention_sink_bias',
            'mlp.gate_proj.weight',
            'mlp.gate_proj.weight_scale_inv',
            'mlp.up_proj.weight',
            'mlp.up_proj.weight_scale_inv',
            'mlp.down_proj.weight',
            'mlp.down_proj.weight_scale_inv',
        }
    )

    def __init__(
        self,
        config: Any,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = MiMoV2FlashMultiTokenPredictor(
            config,
            dtype=dtype,
            device=device,
            prefix='model',
        )
        self._load_buffers: dict[str, dict[str, torch.Tensor]] = {}

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Bind the target model's shared token embedding."""
        self.model.set_input_embeddings(embed_tokens)

    def get_input_embeddings(self):
        """Return the target-owned shared token embedding."""
        return self.model.get_input_embeddings()

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the active prediction depth's final norm before lm_head."""
        return self.model.prepare_hidden_states_for_logits(hidden_states, spec_step_idx=spec_step_idx)

    @staticmethod
    def prepare_hidden_states_for_next_step(
        hidden_states: torch.Tensor,
        logits_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Carry MiMo's decoder output before final norm to the next depth."""
        del logits_hidden_states
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run one same-position MiMo MTP prediction depth."""
        return self.model(
            input_ids,
            position_ids,
            target_hidden_states,
            past_key_values,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            spec_step_idx=spec_step_idx,
        )

    @classmethod
    def _parse_mtp_name(cls, name: str) -> tuple[int, str] | None:
        match = re.fullmatch(r'model\.mtp\.layers\.(\d+)\.(.+)', name)
        if match is None:
            return None
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        if layer_idx not in range(3):
            raise KeyError(f'Unexpected MiMo MTP layer index in {name!r}.')
        if suffix not in cls._MTP_TENSOR_SUFFIXES:
            raise KeyError(f'Unknown MiMo MTP tensor {name!r}.')
        return layer_idx, suffix

    @staticmethod
    def _target_name(layer_idx: int, suffix: str) -> str:
        prefix = f'model.layers.{layer_idx}'
        if suffix in {'enorm.weight', 'hnorm.weight', 'eh_proj.weight', 'final_layernorm.weight'}:
            return f'{prefix}.{suffix}'
        if suffix == 'pre_mlp_layernorm.weight':
            suffix = 'post_attention_layernorm.weight'
        return f'{prefix}.{suffix}'

    def _load_qkv_weight(
        self,
        target_name: str,
        source_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        shard_id: str,
    ):
        tensor_kind = 'scale' if source_name.endswith('.weight_scale_inv') else 'weight'
        source_prefix = source_name.removesuffix('.weight_scale_inv').removesuffix('.weight')
        target_prefix = re.sub(r'\.(q|k|v)_proj$', '.qkv_proj', target_name.rsplit('.', 1)[0])
        target_param = params_dict[f'{target_prefix}.weight']
        buffer = self._load_buffers.setdefault(source_prefix, {})
        buffer[tensor_kind] = loaded_weight
        if set(buffer) != {'weight', 'scale'}:
            return
        weight = _dequantize_blocked_fp8(buffer['weight'], buffer['scale'], target_param.dtype)
        load_weight(target_param, weight, shard_id=shard_id)
        self._load_buffers.pop(source_prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load exactly the three MTP depths and reject incomplete payloads."""
        params_dict = dict(self.named_parameters())
        seen = set()
        for name, loaded_weight in weights:
            parsed = self._parse_mtp_name(name)
            if parsed is None:
                continue
            if name in seen:
                raise KeyError(f'Duplicate MiMo MTP tensor {name!r}.')
            seen.add(name)
            layer_idx, suffix = parsed
            target_name = self._target_name(layer_idx, suffix)

            qkv_match = re.search(r'self_attn\.(q|k|v)_proj\.', suffix)
            if qkv_match is not None:
                self._load_qkv_weight(
                    target_name,
                    name,
                    loaded_weight,
                    params_dict,
                    qkv_match.group(1),
                )
                continue

            for source_projection, target_projection, shard_id in (
                ('gate_proj', 'gate_up_proj', 0),
                ('up_proj', 'gate_up_proj', 1),
            ):
                if f'mlp.{source_projection}.' in suffix:
                    target_name = target_name.replace(source_projection, target_projection)
                    load_weight(params_dict[target_name], loaded_weight, shard_id=shard_id)
                    break
            else:
                load_weight(params_dict[target_name], loaded_weight)

        expected = {
            f'model.mtp.layers.{layer_idx}.{suffix}' for layer_idx in range(3) for suffix in self._MTP_TENSOR_SUFFIXES
        }
        missing = sorted(expected - seen)
        if missing:
            raise KeyError(f'Missing {len(missing)} MiMo MTP tensors; first missing tensor: {missing[0]!r}.')
        if self._load_buffers:
            raise RuntimeError(f'Incomplete MiMo MTP FP8 QKV pairs: {sorted(self._load_buffers)}.')
