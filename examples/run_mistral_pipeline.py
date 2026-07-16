import argparse

from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline


DEFAULT_MODEL_PATH = (
    '/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/models--mistralai--Mistral-7B-v0.1'
)


def print_responses(title, responses):
    print(f'\n========== {title} ==========')
    for i, resp in enumerate(responses):
        print(f'\n===== response {i} =====')
        print(resp.text)


def run_batch_prompt(pipe, gen_config):
    prompts = [
        'Hello, my name is',
        'The capital of France is',
    ]
    responses = pipe(prompts, gen_config=gen_config)
    print_responses('batch string prompts', responses)


def run_openai_messages(pipe, gen_config):
    prompts = [
        [{
            'role': 'user',
            'content': 'Please introduce yourself briefly.',
        }],
        [{
            'role': 'user',
            'content': 'Write one sentence about Paris.',
        }],
    ]
    responses = pipe(prompts, gen_config=gen_config)
    print_responses('OpenAI messages prompts', responses)


def run_stream_prompt(pipe, gen_config):
    prompts = [
        'List three colors:',
        'Complete this sentence: Machine learning is',
    ]

    print('\n========== stream prompts ==========')
    chunks = {}
    for item in pipe.stream_infer(prompts, gen_config=gen_config):
        chunks[item.index] = chunks.get(item.index, '') + item.text
        if item.text:
            print(f'[index={item.index}] {item.text!r}')
        if item.finish_reason:
            print(f'[index={item.index}] finish_reason={item.finish_reason}')

    print('\n========== stream final text ==========')
    for index in sorted(chunks):
        print(f'\n===== response {index} =====')
        print(chunks[index])


def run_chat_session(pipe, gen_config):
    print('\n========== chat session ==========')
    session = pipe.chat('你好，我叫小明。请只回复一句话确认你记住了。', gen_config=gen_config)
    print('\n===== round 1 =====')
    print(session.response.text)
    print(f'session after round 1: {session}')

    session = pipe.chat('我叫什么名字？', session=session, gen_config=gen_config)
    print('\n===== round 2 =====')
    print(session.response.text)
    print(f'session after round 2: {session}')


def run_stream_chat(pipe, gen_config):
    print('\n========== stream chat ==========')
    chunks = []
    for item in pipe.chat('用一句话解释什么是 KV cache。', gen_config=gen_config, stream_response=True):
        chunks.append(item.text)
        if item.text:
            print(item.text, end='', flush=True)
        if item.finish_reason:
            print(f'\nfinish_reason={item.finish_reason}')

    print('\n===== stream chat final text =====')
    print(''.join(chunks))


def parse_args():
    parser = argparse.ArgumentParser(description='Smoke test LMDeploy pipeline APIs.')
    parser.add_argument('--model-path', default=DEFAULT_MODEL_PATH, help='Local model path or HuggingFace repo id.')
    parser.add_argument('--backend', choices=['pytorch', 'turbomind'], default='pytorch', help='Inference backend.')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel size.')
    parser.add_argument('--session-len', type=int, default=4096, help='Max session length.')
    parser.add_argument('--cache-max-entry-count', type=float, default=0.5, help='KV cache memory ratio.')
    parser.add_argument('--max-new-tokens', type=int, default=64, help='Max generated tokens per request.')
    parser.add_argument('--log-level', default='WARNING', help='LMDeploy log level.')
    return parser.parse_args()


def build_backend_config(args):
    config_cls = PytorchEngineConfig if args.backend == 'pytorch' else TurbomindEngineConfig
    return config_cls(
        tp=args.tp,
        session_len=args.session_len,
        cache_max_entry_count=args.cache_max_entry_count,
    )


def main():
    args = parse_args()
    backend_config = build_backend_config(args)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
    )

    print(f'model_path: {args.model_path}')
    print(f'backend_config: {backend_config}')
    print(f'gen_config: {gen_config}')

    with pipeline(args.model_path, backend_config=backend_config, log_level=args.log_level) as pipe:
        run_batch_prompt(pipe, gen_config)
        run_openai_messages(pipe, gen_config)
        run_stream_prompt(pipe, gen_config)
        run_chat_session(pipe, gen_config)
        run_stream_chat(pipe, gen_config)


if __name__ == '__main__':
    main()
