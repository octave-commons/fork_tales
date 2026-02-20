import time
import os
import torch
import statistics
import numpy as np
from typing import List, Any
from sentence_transformers import SentenceTransformer

# Hardware Config
DEVICES = []
if torch.cuda.is_available():
    DEVICES.append(("cuda", "NVIDIA RTX 4070"))
DEVICES.append(("cpu", "Intel i9 CPU"))

# Model Config - Focus on High Quality Small Models supporting MRL
MODELS = [
    (
        "nomic-ai/nomic-embed-text-v1.5",
        {"trust_remote_code": True, "matryoshka_dim": 768},
    ),
    (
        "Qwen/Qwen3-Embedding-0.6B",
        {"trust_remote_code": True, "matryoshka_dim": 1024},
    ),  # Assuming user wants MRL
    (
        "sentence-transformers/all-MiniLM-L6-v2",
        {"trust_remote_code": False, "matryoshka_dim": 384},
    ),  # Baseline
    (
        "BAAI/bge-m3",
        {"trust_remote_code": False, "matryoshka_dim": 1024},
    ),  # Strong, big
]

# Workload Config
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
] * 10  # 100 sentences


def benchmark_model(
    model_name: str, config: dict, device: str, device_name: str, batch_size: int = 1
):
    print(f"\n--- Benchmarking: {model_name} on {device_name} (Batch={batch_size}) ---")

    try:
        # Load Model
        t0 = time.perf_counter()
        model = SentenceTransformer(
            model_name,
            trust_remote_code=config.get("trust_remote_code", False),
            device=device,
        )
        load_time = (time.perf_counter() - t0) * 1000
        print(f"  Load Time: {load_time:.2f} ms")

        # Prepare Matryoshka Slicing if supported (simulated by truncation)
        full_dim = model.get_sentence_embedding_dimension()
        target_dim = 24
        print(f"  Native Dim: {full_dim} -> Target Dim: {target_dim}")

        # Warmup
        print("  Warming up...")
        for _ in range(3):
            _ = model.encode(
                SENTENCES[:batch_size], batch_size=batch_size, convert_to_numpy=True
            )
            if device == "cuda":
                torch.cuda.synchronize()

        # Latency Test (100 rounds)
        latencies = []
        rounds = 10  # Reduced for faster initial feedback
        print(f"  Running {rounds} rounds...")

        start_total = time.perf_counter()
        for i in range(rounds):
            batch = SENTENCES[
                (i * batch_size) % len(SENTENCES) : ((i + 1) * batch_size)
                % len(SENTENCES)
            ]
            if not batch:
                batch = SENTENCES[:batch_size]

            t_start = time.perf_counter()
            embeddings = model.encode(
                batch, batch_size=batch_size, convert_to_numpy=True
            )

            # Simulate Slicing Overhead (negligible but correct)
            if embeddings.shape[1] > target_dim:
                sliced = embeddings[:, :target_dim]
                # Renormalize (MRL requirement)
                norms = np.linalg.norm(sliced, axis=1, keepdims=True)
                sliced = sliced / norms

            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t_start) * 1000)

        total_time = time.perf_counter() - start_total
        avg_ms = statistics.mean(latencies)
        p95_ms = statistics.quantiles(latencies, n=20)[18]
        throughput = (rounds * batch_size) / total_time

        print(f"  Results:")
        print(f"    Avg Latency: {avg_ms:.2f} ms")
        print(f"    P95 Latency: {p95_ms:.2f} ms")
        print(f"    Throughput:  {throughput:.2f} sent/sec")

        return {
            "model": model_name,
            "device": device_name,
            "batch_size": batch_size,
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "throughput": throughput,
        }

    except Exception as e:
        print(f"  Failed: {e}")
        return {"error": str(e), "model": model_name, "device": device_name}


def main():
    print("=== Direct Hardware Benchmark: CPU vs GPU ===")
    results = []

    # Run one by one to avoid timeouts
    target_models = [
        (
            "nomic-ai/nomic-embed-text-v1.5",
            {"trust_remote_code": True, "matryoshka_dim": 768},
        ),
        # ("Qwen/Qwen3-Embedding-0.6B", {"trust_remote_code": True, "matryoshka_dim": 1024}),
    ]

    for model_name, config in target_models:
        for device, device_name in DEVICES:
            # Latency Test (Batch=1)
            res = benchmark_model(model_name, config, device, device_name, batch_size=1)
            results.append(res)

            # Throughput Test (Batch=32) - GPU shines here
            if device == "cuda":
                res_batch = benchmark_model(
                    model_name, config, device, device_name, batch_size=32
                )
                results.append(res_batch)

    print("\n=== Final Summary ===")
    print(
        f"{'Model':<35} | {'Device':<15} | {'Batch':<5} | {'Avg (ms)':<10} | {'Throughput':<12}"
    )
    print("-" * 90)
    for res in results:
        if "error" in res:
            print(
                f"{res['model']:<35} | {res['device']:<15} | {'-':<5} | {'ERROR':<10} | {res['error']}"
            )
        else:
            print(
                f"{res['model']:<35} | {res['device']:<15} | {res['batch_size']:<5} | {res['avg_ms']:<10.2f} | {res['throughput']:<12.2f}"
            )


if __name__ == "__main__":
    main()
