# Copyright (c) OpenMMLab. All rights reserved.
from typing import Literal

import torch
from torch import distributed as dist


class DeviceCommunicator:
    """Device-neutral collective operations for one process group.

    Workspace creation returns a prepared, caller-owned workspace when available. Collective methods execute with native
    fallback when an optimized implementation cannot handle the input.
    """

    def __init__(self, device_group: dist.ProcessGroup):
        self.device_group = device_group

    def all_reduce_(self, input: torch.Tensor):
        """All-reduce ``input`` in place."""
        dist.all_reduce(input, group=self.device_group)

    def all_gather(self, input: torch.Tensor, *, dim: int = -1,
                   workspace=None, copy_output: bool = True) -> torch.Tensor:
        """Gather 2D payloads without changing their dtype or bits.

        Native multi-rank outputs own their storage. An optimized provider may
        borrow its workspace only with copy_output=False; the caller must then
        finish consuming it on the same execution stream before reuse. Input
        shapes, including empty batches, must agree within this process group.
        """
        if input.ndim != 2 or dim not in (0, -1):
            raise ValueError('all_gather requires a 2D input and dim=0 or dim=-1')
        size = dist.get_world_size(self.device_group)
        if size == 1:
            return input
        input = input.contiguous()
        gathered = input.new_empty(size * input.size(0), input.size(1))
        if input.numel():
            dist.all_gather_into_tensor(gathered, input, group=self.device_group)
        if dim == 0:
            return gathered
        return gathered.view(size, *input.shape).transpose(0, 1).reshape(input.size(0), size * input.size(1))

    def create_all_gather_workspace(self, gathered_width: int, device: torch.device, dtype: torch.dtype,
                                    *, dim: int = -1, reuse_sync: bool = True,
                                    capacity_bytes: int = 64 * 1024 * 1024):
        """Prepare an optional caller-owned provider outside graph capture.

        Native collectives need no persistent gather workspace. Providers must
        agree on eligibility across this group before optional rendezvous, and
        protect previous readers before reusing borrowed storage.
        """
        return None

    def all_to_all_single(self, output: torch.Tensor, input: torch.Tensor):
        """Exchange equally sized rank chunks on this communicator's group."""
        dist.all_to_all_single(output, input, group=self.device_group)

    def reduce_scatter_tensor(self, output: torch.Tensor, input: torch.Tensor):
        """Sum and scatter rank chunks on this communicator's group."""
        dist.reduce_scatter_tensor(output, input, group=self.device_group)

    def close(self):
        """Release group-owned resources; gather workspaces belong to
        callers."""


def build_communicator(cpu_group: dist.ProcessGroup, device_group: dist.ProcessGroup,
                       dist_config, *, group_name: Literal['tp', 'dcp'] = 'tp'):
    """Build the default process-group communicator."""
    return DeviceCommunicator(device_group=device_group)
