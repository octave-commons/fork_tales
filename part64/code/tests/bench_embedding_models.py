import time
import os
import json
import statistics
import subprocess
from typing import Any, Callable, List, Optional
from code.world_web.ai import (
    _embed_text,
    _ollama_embed,
    _openvino_embed,
    _ollama_base_url,
    _eta_mu_resize_vector,
)

# Test Data
SENTENCES = [
    "Hello world.",
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "Artificial intelligence is transforming the way we live and work.",
    "Deep learning models require significant computational resources.",
    "Quantum computing promises to solve problems intractable for classical computers.",
    "The simulation hypothesis suggests that reality is an artificial construct.",
    "Embeddings capture semantic meaning in high-dimensional vector space.",
    "Matryoshka representation learning allows for flexible dimensionality reduction.",
] * 5  # 50 total sentences


def benchmark_embedding(
    name: str,
    embed_fn: Callable[[str], Optional[List[float]]],
    warmup: int = 5,
    rounds: int = 50,
) -> dict[str, Any]:
    print(f"\n--- Benchmarking: {name} ---")

    # Warmup
    print(f"Warming up ({warmup} iters)...")
    try:
        for i in range(warmup):
            res = embed_fn(SENTENCES[i % len(SENTENCES)])
            if not res:
                # print(f"  Warning: Warmup {i} returned None/Empty")
                pass
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {"error": str(e), "model": name}

    # Benchmark
    latencies = []
    print(f"Running benchmark ({rounds} rounds)...")
    start_total = time.perf_counter()

    success_count = 0
    for i in range(rounds):
        text = SENTENCES[i % len(SENTENCES)]
        t0 = time.perf_counter()
        try:
            vec = embed_fn(text)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            if vec:
                latencies.append(dt)
                success_count += 1
                # Verify dimensionality (just once)
                if i == 0:
                    print(f"  > Output dims: {len(vec)}")
                    # Test slicing capability (simulating 24-dim requirement)
                    sliced = _eta_mu_resize_vector(vec, 24)
                    print(f"  > Sliced to 24: {len(sliced)}")
            # else:
            #      print(f"  > Round {i}: No vector returned")
        except Exception as e:
            print(f"  > Error on round {i}: {e}")

    total_time = time.perf_counter() - start_total

    if not latencies:
        return {"error": "All requests failed", "model": name}

    avg_ms = statistics.mean(latencies)
    p95_ms = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
    throughput = success_count / total_time

    print(f"Results for {name}:")
    print(f"  Avg Latency: {avg_ms:.2f} ms")
    print(f"  P95 Latency: {p95_ms:.2f} ms")
    print(f"  Throughput:  {throughput:.2f} req/sec")

    return {"model": name, "avg_ms": avg_ms, "p95_ms": p95_ms, "throughput": throughput}


def main():
    print("=== Embedding Model Benchmark ===")

    # Setup Env Context
    # Try localhost first since we are likely on the host or in a container sharing network
    # Default to localhost:18000 if unset
    if "OPENVINO_EMBED_ENDPOINT" not in os.environ:
        os.environ["OPENVINO_EMBED_ENDPOINT"] = "http://127.0.0.1:18000/v1/embeddings"

    ollama_url = _ollama_base_url()
    openvino_ep = os.getenv("OPENVINO_EMBED_ENDPOINT")
    current_npu_model = os.getenv("OPENVINO_EMBED_MODEL", "nomic-embed-text")

    print(f"Ollama Base URL: {ollama_url}")
    print(f"OpenVINO Endpoint: {openvino_ep}")
    print(f"Current NPU Model (env): {current_npu_model}")

    results = []

    # 1. Benchmark Current NPU Model (likely Nomic) via OpenVINO endpoint
    print("\n[Test 1] Testing configured NPU/OpenVINO endpoint...")
    res_npu = benchmark_embedding(
        f"NPU Current ({current_npu_model})",
        lambda txt: _openvino_embed(txt, model=None),
    )
    results.append(res_npu)

    # 2. Benchmark Qwen3-Embedding-0.6B via Ollama
    qwen_model_name = "qwen3-embedding-0.6b"
    print(f"\n[Setup] Checking for {qwen_model_name}...")

    # Check if model exists in Ollama list
    try:
        list_proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if qwen_model_name not in list_proc.stdout:
            print(
                f"  Model '{qwen_model_name}' not found. Attempting pull (this may take time)..."
            )
            # Try to pull, but might fail if tag doesn't exist
            pull_res = subprocess.run(
                ["ollama", "pull", qwen_model_name], capture_output=True, text=True
            )
            if pull_res.returncode != 0:
                print(f"  Pull failed: {pull_res.stderr.strip()}")
                print(
                    f"  Note: You may need to manually import '{qwen_model_name}' or specify a different tag."
                )
        else:
            print(f"  Model '{qwen_model_name}' found.")
    except Exception as e:
        print(f"  Ollama check failed: {e}")

    print(f"\n[Test 2] Testing {qwen_model_name} on Ollama...")

    def qwen_wrapper(text):
        return _ollama_embed(text, model=qwen_model_name)

    res_qwen = benchmark_embedding(f"Ollama ({qwen_model_name})", qwen_wrapper)
    results.append(res_qwen)

    # 3. Benchmark Nomic via Ollama (Baseline)
    print(f"\n[Test 3] Testing nomic-embed-text on Ollama (Baseline)...")
    res_nomic_ollama = benchmark_embedding(
        "Ollama (nomic-embed-text)",
        lambda txt: _ollama_embed(txt, model="nomic-embed-text"),
    )
    results.append(res_nomic_ollama)

    print("\n=== Final Summary ===")
    print(f"{'Model':<35} | {'Avg (ms)':<10} | {'Throughput':<12} | {'Status'}")
    print("-" * 75)
    for res in results:
        model_name = res.get("model", "Unknown")
        if "error" in res:
            print(f"{model_name:<35} | {'ERROR':<10} | {'-':<12} | {res['error']}")
        else:
            print(
                f"{model_name:<35} | {res['avg_ms']:<10.2f} | {res['throughput']:<12.2f} | OK"
            )


if __name__ == "__main__":
    main()
