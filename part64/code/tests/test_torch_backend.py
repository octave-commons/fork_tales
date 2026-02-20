import os
import time
from code.world_web.ai import _embed_text

# Set backend to torch
os.environ["EMBEDDINGS_BACKEND"] = "torch"
os.environ["TORCH_EMBED_MODEL"] = "nomic-ai/nomic-embed-text-v1.5"

print("--- Testing Torch Backend ---")
t0 = time.perf_counter()
vec = _embed_text("Hello world via torch")
dt = (time.perf_counter() - t0) * 1000

if vec:
    print(f"Success! Vector dim: {len(vec)}")
    print(f"Latency (cold): {dt:.2f} ms")

    t0 = time.perf_counter()
    vec = _embed_text("Hello again")
    dt = (time.perf_counter() - t0) * 1000
    print(f"Latency (warm): {dt:.2f} ms")
else:
    print("Failed to get vector.")
