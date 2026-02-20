# Nexus Daimoi: Semantic Fields & Arbitrary Dimensionality

## Overview
Proposed extension to the Nexus Daimoi simulation engine to support user-defined semantic fields with configurable dimensionality, powered by NPU-accelerated embeddings (e.g., Nomic v1.5).

## Goals
1.  **Arbitrary Semantic Fields**: Allow users to define fields like "Privacy", "Security", "Joy" via text prompts.
2.  **Configurable Dimensionality**: Support arbitrary embedding dimensions (e.g., 64, 128, 768) per field layer, moving beyond the hardcoded 24-dim limit.
3.  **Layering**: Allow multiple fields to overlap and interact.
4.  **UI Configuration**: A "nice UI" for setting up these fields and layers dynamically.

## Architecture

### 1. Engine Refactor (C Double Buffer)
-   Current: `float *embeddings` (24-dim hardcoded).
-   Proposed:
    -   `Particle` struct stores `float *embedding_ptr`.
    -   Global `EmbeddingStore` manages variable-length vectors.
    -   `FieldLayer` struct defines:
        -   `vector_t *anchor_vector` (The semantic meaning of the field)
        -   `float strength`
        -   `float radius`
        -   `int dimension_slice_start` (For Matryoshka slicing support)
        -   `int dimension_slice_end`

### 2. Python Backend (`daimoi_probabilistic.py` & `simulation.py`)
-   Implement `SemanticFieldRegistry` to manage user-defined fields.
-   Integrate `ai._embed_text` with Matryoshka slicing (completed for particles, needed for fields).
-   Expose API endpoints:
    -   `POST /api/simulation/fields` (Create new field)
    -   `GET /api/simulation/fields` (List active fields)
    -   `PATCH /api/simulation/fields/{id}` (Tune strength/dim)

### 3. Frontend UI (`part64/frontend`)
-   **Field Designer**:
    -   Text input for field concept (e.g., "Cybersecurity").
    -   Slider for dimensionality (Trade-off: Precision vs Speed).
    -   Slider for influence strength.
    -   Color picker for visualization.
-   **Layer Manager**:
    -   Drag-and-drop ordering of active fields.
    -   Opacity controls for visualizing field influence.

## Next Steps
1.  (Done) Implement Nomic/Matryoshka support for particle generation.
2.  (Pending) Prototype `SemanticFieldRegistry` in Python.
3.  (Pending) Draft C engine changes for variable-dim support.
4.  (Pending) Build Frontend UI.

## Open Questions
-   Performance impact of >24 dims on 20k+ particles in C engine?
-   How to visualize high-dimensional similarity in 2D? (PCA/t-SNE in real-time or simple projection?)
