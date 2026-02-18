from __future__ import annotations

import argparse
import json
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import yaml

from constitution_runtime import (
    append_registry_row,
    load_manifest_for_dataset,
    now_iso,
    verify_dataset_checksums,
    verify_training_examples,
)


FIELD_ORDER = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
FIELD_NAMES = {
    "f1": "artifact_flux",
    "f2": "witness_tension",
    "f3": "coherence_focus",
    "f4": "drift_pressure",
    "f5": "fork_tax_balance",
    "f6": "curiosity_drive",
    "f7": "gate_pressure",
    "f8": "council_heat",
}
FIELD_INDEX = {field_id: index for index, field_id in enumerate(FIELD_ORDER)}


@dataclass
class DatasetConfig:
    train_jsonl: str
    val_jsonl: Optional[str]
    image_root: str


@dataclass
class ModelConfig:
    image_size: int
    hidden_size: int
    dropout: float


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mixed_precision: bool
    augment: bool
    kl_weight: float
    ce_weight: float
    checkpoint_dir: str
    export_dir: str
    log_dir: str
    max_train_samples: int
    max_val_samples: int


@dataclass
class Config:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


@dataclass
class ImageExample:
    image_path: str
    target_distribution: np.ndarray
    dominant_label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulation-optimized image training loop for Promethean fields"
    )
    parser.add_argument("--config", type=str, default="/workspace/config.image.yaml")
    return parser.parse_args()


def read_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    dataset = DatasetConfig(
        train_jsonl=payload["dataset"]["train_jsonl"],
        val_jsonl=payload["dataset"].get("val_jsonl"),
        image_root=payload["dataset"]["image_root"],
    )

    model = ModelConfig(
        image_size=int(payload["model"]["image_size"]),
        hidden_size=int(payload["model"]["hidden_size"]),
        dropout=float(payload["model"].get("dropout", 0.2)),
    )

    training = TrainingConfig(
        epochs=int(payload["training"]["epochs"]),
        batch_size=int(payload["training"]["batch_size"]),
        learning_rate=float(payload["training"]["learning_rate"]),
        weight_decay=float(payload["training"]["weight_decay"]),
        mixed_precision=bool(payload["training"].get("mixed_precision", False)),
        augment=bool(payload["training"].get("augment", True)),
        kl_weight=float(payload["training"].get("kl_weight", 0.65)),
        ce_weight=float(payload["training"].get("ce_weight", 0.35)),
        checkpoint_dir=payload["training"]["checkpoint_dir"],
        export_dir=payload["training"]["export_dir"],
        log_dir=payload["training"]["log_dir"],
        max_train_samples=int(payload["training"].get("max_train_samples", 0)),
        max_val_samples=int(payload["training"].get("max_val_samples", 0)),
    )

    return Config(
        seed=int(payload.get("seed", 42)),
        dataset=dataset,
        model=model,
        training=training,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or not path.is_file():
        return []

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def normalize_scores(
    scores: Dict[str, float], fallback_field: str = "f6"
) -> np.ndarray:
    values = np.zeros((len(FIELD_ORDER),), dtype=np.float32)
    for index, field_id in enumerate(FIELD_ORDER):
        raw = float(scores.get(field_id, 0.0))
        values[index] = max(0.0, raw)

    total = float(values.sum())
    if total <= 0.0:
        fallback = FIELD_INDEX.get(fallback_field, FIELD_INDEX["f6"])
        values[:] = 0.0
        values[fallback] = 1.0
        return values

    return values / total


def parse_response_payload(row: Dict[str, object]) -> Dict[str, object]:
    payload = row.get("response")
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    elif isinstance(payload, dict):
        return payload
    return {}


def resolve_image_path(row: Dict[str, object], image_root: Path) -> Optional[Path]:
    image_value = row.get("image")
    if not isinstance(image_value, str) or not image_value.strip():
        return None
    candidate = Path(image_value)
    if not candidate.is_absolute():
        candidate = image_root / candidate
    candidate = candidate.resolve()
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def row_to_example(row: Dict[str, object], image_root: Path) -> Optional[ImageExample]:
    image_path = resolve_image_path(row, image_root)
    if image_path is None:
        return None

    payload = parse_response_payload(row)
    metadata = row.get("metadata")
    metadata_map = metadata if isinstance(metadata, dict) else {}

    field_scores_payload = payload.get("field_scores")
    if isinstance(field_scores_payload, dict):
        score_map = {
            field_id: float(field_scores_payload.get(field_id, 0.0))
            for field_id in FIELD_ORDER
        }
    else:
        dominant_field = str(
            payload.get("dominant_field") or metadata_map.get("dominant_field") or "f6"
        )
        score_map = {field_id: 0.0 for field_id in FIELD_ORDER}
        score_map[dominant_field if dominant_field in score_map else "f6"] = 1.0

    target_distribution = normalize_scores(score_map)
    dominant_label = int(np.argmax(target_distribution).item())

    return ImageExample(
        image_path=str(image_path),
        target_distribution=target_distribution,
        dominant_label=dominant_label,
    )


def load_image_examples(
    *,
    path: Path,
    image_root: Path,
    max_samples: int,
    seed: int,
) -> List[ImageExample]:
    rows = read_jsonl(path)
    examples: List[ImageExample] = []
    for row in rows:
        example = row_to_example(row, image_root=image_root)
        if example is not None:
            examples.append(example)

    rng = random.Random(seed)
    rng.shuffle(examples)

    if max_samples > 0:
        examples = examples[:max_samples]

    return examples


def load_image_tensor(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array


def make_generator(
    examples: Iterable[ImageExample], image_size: int
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.int32]]:
    for example in examples:
        image = load_image_tensor(example.image_path, image_size=image_size)
        label = np.int32(example.dominant_label)
        yield image, example.target_distribution.astype(np.float32), label


def build_dataset(
    *,
    examples: List[ImageExample],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(len(FIELD_ORDER),), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: make_generator(examples, image_size=image_size),
        output_signature=output_signature,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(32, len(examples)), seed=seed)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_image_router_model(
    *, image_size: int, hidden_size: int, dropout: float
) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(image_size, image_size, 3), name="pixel_values", dtype=tf.float32
    )

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="gelu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(384, 3, padding="same", activation="gelu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(hidden_size, activation="gelu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(max(128, hidden_size // 2), activation="gelu")(x)

    logits = tf.keras.layers.Dense(
        len(FIELD_ORDER),
        name="field_logits",
        dtype="float32",
    )(x)

    return tf.keras.Model(
        inputs=inputs, outputs=logits, name="simulation_image_field_router"
    )


def compute_class_weights(examples: List[ImageExample]) -> np.ndarray:
    counts = np.ones((len(FIELD_ORDER),), dtype=np.float32)
    for example in examples:
        counts[example.dominant_label] += 1.0

    weights = counts.sum() / (len(FIELD_ORDER) * counts)
    weights /= weights.mean()
    return weights.astype(np.float32)


def latest_weight_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoints = sorted(checkpoint_dir.glob("epoch-*.weights.h5"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def augment_batch(images: tf.Tensor) -> tf.Tensor:
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.08)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    images = tf.clip_by_value(images, 0.0, 1.0)
    return images


def run_training(cfg: Config) -> None:
    if cfg.training.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    train_examples = load_image_examples(
        path=Path(cfg.dataset.train_jsonl),
        image_root=Path(cfg.dataset.image_root),
        max_samples=cfg.training.max_train_samples,
        seed=cfg.seed,
    )
    if not train_examples:
        raise ValueError("No image samples found in training dataset")

    val_examples: List[ImageExample] = []
    if cfg.dataset.val_jsonl:
        val_path = Path(cfg.dataset.val_jsonl)
        if val_path.exists():
            val_examples = load_image_examples(
                path=val_path,
                image_root=Path(cfg.dataset.image_root),
                max_samples=cfg.training.max_val_samples,
                seed=cfg.seed + 1,
            )

    train_ds = build_dataset(
        examples=train_examples,
        image_size=cfg.model.image_size,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    val_ds = (
        build_dataset(
            examples=val_examples,
            image_size=cfg.model.image_size,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            seed=cfg.seed,
        )
        if val_examples
        else None
    )

    model = build_image_router_model(
        image_size=cfg.model.image_size,
        hidden_size=cfg.model.hidden_size,
        dropout=cfg.model.dropout,
    )

    _ = model(
        tf.zeros((1, cfg.model.image_size, cfg.model.image_size, 3), dtype=tf.float32),
        training=False,
    )

    class_weights = tf.constant(compute_class_weights(train_examples), dtype=tf.float32)

    base_optimizer = tf.keras.optimizers.AdamW(
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer = (
        tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        if cfg.training.mixed_precision
        else base_optimizer
    )

    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    export_dir = Path(cfg.training.export_dir)
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = latest_weight_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        model.load_weights(str(latest_ckpt))
        print(f"Resumed from checkpoint: {latest_ckpt}")

    summary_writer = tf.summary.create_file_writer(str(log_dir))
    global_step = 0

    for epoch in range(cfg.training.epochs):
        print(f"\nImage epoch {epoch + 1}/{cfg.training.epochs}")

        train_loss_metric = tf.keras.metrics.Mean()
        train_acc_metric = tf.keras.metrics.Mean()
        train_kl_metric = tf.keras.metrics.Mean()
        train_ce_metric = tf.keras.metrics.Mean()

        for images, target_dist, labels in tqdm(
            train_ds, desc="image-train", leave=False
        ):
            if cfg.training.augment:
                images = augment_batch(images)

            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                probs = tf.nn.softmax(logits, axis=-1)

                kl = tf.keras.losses.kullback_leibler_divergence(target_dist, probs)
                ce = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
                sample_weights = tf.gather(class_weights, labels)
                weighted_ce = ce * sample_weights

                kl_mean = tf.reduce_mean(kl)
                ce_mean = tf.reduce_mean(weighted_ce)
                loss = (cfg.training.kl_weight * kl_mean) + (
                    cfg.training.ce_weight * ce_mean
                )

                if cfg.training.mixed_precision:
                    scaled_loss = optimizer.get_scaled_loss(loss)
                else:
                    scaled_loss = loss

            gradients = tape.gradient(scaled_loss, model.trainable_variables)
            if cfg.training.mixed_precision:
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(predictions, labels), tf.float32)
            )

            train_loss_metric.update_state(loss)
            train_acc_metric.update_state(accuracy)
            train_kl_metric.update_state(kl_mean)
            train_ce_metric.update_state(ce_mean)
            global_step += 1

            if global_step % 10 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "image_train/loss", train_loss_metric.result(), step=global_step
                    )
                    tf.summary.scalar(
                        "image_train/accuracy",
                        train_acc_metric.result(),
                        step=global_step,
                    )
                    tf.summary.scalar(
                        "image_train/kl", train_kl_metric.result(), step=global_step
                    )
                    tf.summary.scalar(
                        "image_train/ce", train_ce_metric.result(), step=global_step
                    )

        print(
            "train_loss={:.6f} train_acc={:.4f} train_kl={:.6f} train_ce={:.6f}".format(
                float(train_loss_metric.result()),
                float(train_acc_metric.result()),
                float(train_kl_metric.result()),
                float(train_ce_metric.result()),
            )
        )

        if val_ds is not None:
            val_loss_metric = tf.keras.metrics.Mean()
            val_acc_metric = tf.keras.metrics.Mean()
            for images, target_dist, labels in tqdm(
                val_ds, desc="image-val", leave=False
            ):
                logits = model(images, training=False)
                probs = tf.nn.softmax(logits, axis=-1)

                kl = tf.keras.losses.kullback_leibler_divergence(target_dist, probs)
                ce = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
                sample_weights = tf.gather(class_weights, labels)
                weighted_ce = ce * sample_weights
                loss = (cfg.training.kl_weight * tf.reduce_mean(kl)) + (
                    cfg.training.ce_weight * tf.reduce_mean(weighted_ce)
                )

                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(predictions, labels), tf.float32)
                )

                val_loss_metric.update_state(loss)
                val_acc_metric.update_state(accuracy)

            with summary_writer.as_default():
                tf.summary.scalar(
                    "image_val/loss", val_loss_metric.result(), step=epoch + 1
                )
                tf.summary.scalar(
                    "image_val/accuracy", val_acc_metric.result(), step=epoch + 1
                )

            print(
                "val_loss={:.6f} val_acc={:.4f}".format(
                    float(val_loss_metric.result()),
                    float(val_acc_metric.result()),
                )
            )

        ckpt_path = (
            checkpoint_dir / f"epoch-{epoch + 1:04d}-step-{global_step:08d}.weights.h5"
        )
        model.save_weights(str(ckpt_path))
        print(f"image_checkpoint={ckpt_path}")

    if export_dir.exists():
        shutil.rmtree(export_dir)
    tf.saved_model.save(model, str(export_dir))

    meta = {
        "field_order": FIELD_ORDER,
        "field_names": FIELD_NAMES,
        "class_weights": class_weights.numpy().tolist(),
        "image_size": cfg.model.image_size,
    }
    (export_dir / "field_router_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved image router export to {export_dir}")


def main() -> None:
    args = parse_args()
    cfg = read_config(Path(args.config))
    set_seed(cfg.seed)
    run_training(cfg)


if __name__ == "__main__":
    main()
