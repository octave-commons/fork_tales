"""TensorFlow model stub for Qwen3-VL fine-tuning harness.

Replace `build_model` with your actual Qwen3-VL:2B TensorFlow implementation.
This reference model is intentionally small and only validates the training stack.
"""

from __future__ import annotations

from typing import Dict, Optional

import tensorflow as tf


class _ReferenceVisionLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__(name="reference_vision_language_model")
        self.token_embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name="token_embedding",
        )

        self.vision_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(64, 3, activation="gelu", padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(128, 3, activation="gelu", padding="same"),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Conv2D(256, 3, activation="gelu", padding="same"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(hidden_size, activation="gelu"),
            ],
            name="vision_encoder",
        )

        self.self_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=hidden_size // 8,
            dropout=0.1,
            name="self_attention",
        )
        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_size * 4, activation="gelu"),
                tf.keras.layers.Dense(hidden_size),
            ],
            name="ffn",
        )

        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.output_head = tf.keras.layers.Dense(vocab_size, name="lm_head")

    def call(
        self,
        inputs,
        training: Optional[bool] = None,
        mask=None,
    ) -> tf.Tensor:
        del mask
        features: Dict[str, tf.Tensor] = inputs
        training_flag = bool(training)

        input_ids = features["input_ids"]
        pixel_values = features["pixel_values"]

        token_states = self.token_embedding(input_ids)
        visual_state = self.vision_encoder(pixel_values, training=training_flag)
        visual_state = tf.expand_dims(visual_state, axis=1)

        fused = token_states + visual_state

        attn_out = self.self_attention(
            query=fused,
            value=fused,
            key=fused,
            training=training_flag,
        )
        fused = self.norm1(fused + attn_out)
        fused = self.norm2(fused + self.ffn(fused, training=training_flag))

        return self.output_head(fused)


def build_model(vocab_size: int, hidden_size: int) -> tf.keras.Model:
    """Build trainable model graph.

    Swap this with your real TensorFlow Qwen3-VL:2B port while keeping
    the same call signature (`features -> logits`).
    """

    return _ReferenceVisionLanguageModel(vocab_size=vocab_size, hidden_size=hidden_size)
