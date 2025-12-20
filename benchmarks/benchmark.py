import torch

import time

def generate_with_metrics(model : torch.nn.Module, tokenizer, prompt, max_new_tokens=128):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(next(model.parameters()).device)  
    
    prefill_start = time.monotonic()
    with torch.no_grad():
        logits = model(tokens)
    prefill_time = time.monotonic() - prefill_start
    ttft = prefill_time 

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    tokens = torch.cat((tokens, next_token), dim=1)

    decode_start = time.monotonic()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            logits = model(tokens)
            
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        tokens = torch.cat((tokens, next_token), dim=1)
    
    decode_time = time.monotonic() - decode_start
    total_time = prefill_time + decode_time
    
    decoded = tokenizer.decode(tokens.squeeze(0).tolist())
    
    metrics = {
        "prefill_time": prefill_time,
        "ttft": ttft,
        "decode_time": decode_time,
        "total_time": total_time,
        "gen_len": max_new_tokens
    }
    
    return decoded, metrics

def benchmark_generation(model, tokenizer, prompt, warmup=1, iters=5, max_new_tokens=20):
    print(f"Warming up {warmup} times...")
    for _ in range(warmup):
        _ = generate_with_metrics(model, tokenizer, prompt, max_new_tokens=10)

    stats = {
        "ttft": 0,
        "prefill": 0,
        "tps": 0, 
        "latency": 0
    }

    for i in range(iters):
        out, m = generate_with_metrics(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        
        tps = m['gen_len'] / m['total_time']
        stats["ttft"] += m['ttft']
        stats["prefill"] += m['prefill_time']
        stats["tps"] += tps
        stats["latency"] += m['total_time']

        print(f"Iter {i+1}: TTFT: {m['ttft']:.4f}s | Prefill: {m['prefill_time']:.4f}s | TPS: {tps:.2f}")

    print("\n--- Final Results ---")
    print(f"Avg TTFT: {stats['ttft']/iters:.4f}s")
    print(f"Avg Prefill: {stats['prefill']/iters:.4f}s")
    print(f"Avg Tokens/s: {stats['tps']/iters:.2f}")
    print(f"Avg Latency: {stats['latency']/iters:.4f}s")

    return stats

if __name__ == "__main__":

    from llm.qwen3.qwen_torch import Qwen3
    from llm.qwen3.fast_qwen_cuda import FastQwen3
    from llm.qwen3.qwen_token import Qwen3Tokenizer
    from llm.qwen3.config import QwenConfig_bfloat16 , QwenConfig_float16

    device = torch.device("cuda")
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    config_qwen_bf16 =QwenConfig_bfloat16()
    config_qwen_fp16 = QwenConfig_float16()
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )

    model_torch = Qwen3(config_qwen_bf16).to(device)
    model_fast = FastQwen3(config_qwen_fp16).to(device)


    max_new_token_range = [32 , 64 , 128 , 256 , 512 , 1024]

    latency = []
    token_per_sec = []

    print("Start benchmarking")
    for token_size in max_new_token_range:
        pass


    
