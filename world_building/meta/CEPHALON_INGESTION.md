# Cephalon Ingestion Recipe (offline)

This is a minimal, provider-agnostic ingestion plan.

## 1) Parse
- Walk `operation-mindfuck/` excluding `source_archives/` and `search/`.
- For text: UTF-8 decode with `errors=ignore`.
- For PDFs: extract text if possible.
- For images: index metadata + path; optional OCR later.

## 2) Normalize
For each artifact:
- `sha256`, `bytes`, `mime`, `path`
- `circuit_scores` (8)
- `vector64` (64)

## 3) Store
- Put the artifact row into Mongo (or your canonical DB).
- Put a retrieval index into:
  - SQLite FTS (already included) OR
  - your vector DB (Chroma/etc.) using embeddings later

## 4) Retrieve
- First pass: FTS query.
- Second pass: circuit filter (top circuit or threshold).
- Third pass: nearest neighbors (vector64) to expand context.

## 5) Upgrade path
Swap `vector64` with:
- per-circuit embeddings (8 embedding models) or
- one embedding model + project into 8 layers

Keep `vector64` around as a stable compatibility layer.
