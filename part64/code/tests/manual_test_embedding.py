from code.world_web.daimoi_probabilistic import _embedding_from_text, DAIMOI_EMBED_DIMS
from code.world_web.ai import _embed_text
import time

print(f"Default dims: {DAIMOI_EMBED_DIMS}")

# Test 1: Basic hashing (fallback)
# If NPU/Ollama is not running, this will fallback to hash
vec1 = _embedding_from_text("hello world")
print(f"Vec1 len: {len(vec1)}")
print(f"Vec1 norm: {sum(x * x for x in vec1)}")

# Test 2: Semantic (if mockable)
# We can mock ai._embed_text to simulate NPU response
import sys
from unittest.mock import MagicMock

# Mock the _embed_text function in daimoi_probabilistic's namespace (which imported it locally)
# Actually, since it imports inside the function, we need to mock code.world_web.ai._embed_text
import code.world_web.ai

code.world_web.ai._embed_text = MagicMock(return_value=[0.1] * 768)

# Clear cache to force re-run
from code.world_web.daimoi_probabilistic import _semantic_embedding_cached

_semantic_embedding_cached.cache_clear()

vec2 = _embedding_from_text("semantic test")
print(f"Vec2 len: {len(vec2)}")
print(f"Vec2 first 5: {vec2[:5]}")

# Verify slicing happened (768 -> 24)
assert len(vec2) == 24
print("Slicing verified.")
