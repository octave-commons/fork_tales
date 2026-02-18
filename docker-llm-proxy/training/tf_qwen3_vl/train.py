from __future__ import annotations

import argparse
import json
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import yaml
from transformers import AutoTokenizer

from constitution_runtime import (
    append_registry_row,
    load_manifest_for_dataset,
    now_iso,
    verify_dataset_checksums,
    verify_training_examples,
)
from model_impl import build_model


@dataclass
class DatasetConfig:
    train_jsonl: str
    val_jsonl: Optional[str]
    image_root: str


@dataclass
class ModelConfig:
    tokenizer_id: str
    max_seq_len: int
    image_size: int
    hidden_size: int


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    mixed_precision: bool
    checkpoint_dir: str
    export_dir: str
    log_dir: str


@dataclass
class Config:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig


def read_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    dataset = DatasetConfig(
        train_jsonl=payload["dataset"]["train_jsonl"],
        val_jsonl=payload["dataset"].get("val_jsonl"),
        image_root=payload["dataset"]["image_root"],
    )

    model = ModelConfig(
        tokenizer_id=payload["model"]["tokenizer_id"],
        max_seq_len=int(payload["model"]["max_seq_len"]),
        image_size=int(payload["model"]["image_size"]),
        hidden_size=int(payload["model"]["hidden_size"]),
    )

    training = TrainingConfig(
        epochs=int(payload["training"]["epochs"]),
        batch_size=int(payload["training"]["batch_size"]),
        gradient_accumulation_steps=int(
            payload["training"]["gradient_accumulation_steps"]
        ),
        learning_rate=float(payload["training"]["learning_rate"]),
        weight_decay=float(payload["training"]["weight_decay"]),
        mixed_precision=bool(payload["training"]["mixed_precision"]),
        checkpoint_dir=payload["training"]["checkpoint_dir"],
        export_dir=payload["training"]["export_dir"],
        log_dir=payload["training"]["log_dir"],
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


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).lower()
                if item_type == "text":
                    chunks.append(str(item.get("text", "")))
                elif "image" in item_type:
                    chunks.append("<image>")
                elif "url" in item_type:
                    chunks.append(str(item.get("url", "")))
            else:
                chunks.append(str(item))
        return "\n".join(chunk for chunk in chunks if chunk)

    return str(content)


def find_image_path(record: Dict[str, object]) -> Optional[str]:
    if isinstance(record.get("image"), str):
        return str(record["image"])

    messages = record.get("messages")
    if not isinstance(messages, list):
        return None

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if "image" not in str(item.get("type", "")).lower():
                    continue
                if isinstance(item.get("path"), str):
                    return str(item["path"])
                image_url = item.get("image_url")
                if isinstance(image_url, str):
                    return image_url
                if isinstance(image_url, dict) and isinstance(
                    image_url.get("url"), str
                ):
                    return str(image_url["url"])

    return None


def extract_prompt_response(record: Dict[str, object]) -> Tuple[str, str]:
    if isinstance(record.get("prompt"), str) and isinstance(
        record.get("response"), str
    ):
        return str(record["prompt"]), str(record["response"])

    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Each record needs prompt/response or messages")

    prompt_parts: List[str] = []
    response = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "user")).lower()
        content = normalize_message_content(msg.get("content", ""))
        if role == "assistant":
            response = content
        else:
            prompt_parts.append(f"{role}: {content}")

    prompt = "\n".join(prompt_parts).strip()
    if not prompt or not response:
        raise ValueError("Could not derive prompt/response from record messages")

    return prompt, response


def load_records(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {index} in {path}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {index} in {path} is not a JSON object")
            rows.append(row)

    if not rows:
        raise ValueError(f"No training records found in {path}")

    return rows


def resolve_image(
    image_root: Path, image_ref: Optional[str], image_size: int
) -> np.ndarray:
    if not image_ref:
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

    image_path = Path(image_ref)
    if not image_path.is_absolute():
        image_path = image_root / image_path

    if not image_path.exists():
        return np.zeros((image_size, image_size, 3), dtype=np.float32)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0

    return array


def encode_example(
    tokenizer,
    prompt: str,
    response: str,
    max_seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    eos_id = (
        tokenizer.eos_token_id
        if tokenizer.eos_token_id is not None
        else tokenizer.pad_token_id
    )

    if eos_id is None:
        eos_id = 0

    input_ids = prompt_ids + response_ids + [eos_id]
    labels = ([-100] * len(prompt_ids)) + response_ids + [eos_id]

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]

    attention_mask = [1] * len(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    while len(input_ids) < max_seq_len:
        input_ids.append(pad_id)
        attention_mask.append(0)
        labels.append(-100)

    return (
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(attention_mask, dtype=np.int32),
        np.asarray(labels, dtype=np.int32),
    )


def make_generator(
    records: Iterable[Dict[str, object]],
    tokenizer,
    max_seq_len: int,
    image_root: Path,
    image_size: int,
) -> Iterator[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    for row in records:
        prompt, response = extract_prompt_response(row)
        image_ref = find_image_path(row)

        input_ids, attention_mask, labels = encode_example(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            max_seq_len=max_seq_len,
        )
        pixel_values = resolve_image(
            image_root=image_root,
            image_ref=image_ref,
            image_size=image_size,
        )

        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        yield features, labels


def build_dataset(
    records: List[Dict[str, object]],
    tokenizer,
    cfg: Config,
    shuffle: bool,
) -> tf.data.Dataset:
    image_root = Path(cfg.dataset.image_root)
    output_signature = (
        {
            "input_ids": tf.TensorSpec(shape=(cfg.model.max_seq_len,), dtype=tf.int32),
            "attention_mask": tf.TensorSpec(
                shape=(cfg.model.max_seq_len,), dtype=tf.int32
            ),
            "pixel_values": tf.TensorSpec(
                shape=(cfg.model.image_size, cfg.model.image_size, 3), dtype=tf.float32
            ),
        },
        tf.TensorSpec(shape=(cfg.model.max_seq_len,), dtype=tf.int32),
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: make_generator(
            records=records,
            tokenizer=tokenizer,
            max_seq_len=cfg.model.max_seq_len,
            image_root=image_root,
            image_size=cfg.model.image_size,
        ),
        output_signature=output_signature,
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(32, len(records)), seed=cfg.seed)

    dataset = dataset.batch(cfg.training.batch_size, drop_remainder=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def masked_cross_entropy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    mask = tf.not_equal(labels, -100)
    safe_labels = tf.where(mask, labels, tf.zeros_like(labels))

    losses = tf.keras.losses.sparse_categorical_crossentropy(
        y_true=safe_labels,
        y_pred=logits,
        from_logits=True,
    )
    mask_f = tf.cast(mask, losses.dtype)

    numerator = tf.reduce_sum(losses * mask_f)
    denominator = tf.maximum(tf.reduce_sum(mask_f), 1.0)
    return numerator / denominator


def latest_weight_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoints = sorted(checkpoint_dir.glob("epoch-*.weights.h5"))
    if not checkpoints:
        return None
    return checkpoints[-1]


def run_training(cfg: Config, *, config_path: Path) -> None:
    if cfg.model.tokenizer_id.startswith("REPLACE_WITH"):
        raise ValueError("Set model.tokenizer_id in config.yaml before training")

    if cfg.training.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_id, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_jsonl_path = Path(cfg.dataset.train_jsonl).resolve()
    manifest_path, manifest = load_manifest_for_dataset(train_jsonl_path)
    checksum_summary = verify_dataset_checksums(manifest)
    constitution_id = str(manifest.get("constitution_id", "")).strip() or None

    examples_path_value = manifest.get("training_examples_path")
    if not isinstance(examples_path_value, str) or not examples_path_value.strip():
        examples_path_value = str(
            train_jsonl_path.with_name("training_examples.v1.jsonl")
        )
    training_examples_summary = verify_training_examples(
        Path(examples_path_value),
        expected_taxonomy=constitution_id,
    )
    print(
        json.dumps(
            {
                "preflight": "constitution",
                "constitution_id": constitution_id,
                "manifest_path": str(manifest_path),
                "dataset_checksums": checksum_summary,
                "training_examples": training_examples_summary,
            },
            ensure_ascii=False,
        )
    )

    train_records = load_records(Path(cfg.dataset.train_jsonl))
    if not cfg.dataset.val_jsonl:
        raise ValueError(
            "Constitution layer-8 eval-required: dataset.val_jsonl must be configured"
        )

    val_path = Path(cfg.dataset.val_jsonl)
    if not val_path.exists() or not val_path.is_file():
        raise FileNotFoundError(
            f"Constitution layer-8 eval-required: val_jsonl file not found: {val_path}"
        )

    val_records: List[Dict[str, object]] = load_records(val_path)

    train_ds = build_dataset(train_records, tokenizer, cfg, shuffle=True)
    val_ds = (
        build_dataset(val_records, tokenizer, cfg, shuffle=False)
        if val_records
        else None
    )

    model = build_model(
        vocab_size=int(len(tokenizer)),
        hidden_size=cfg.model.hidden_size,
    )

    _ = model(
        {
            "input_ids": tf.zeros((1, cfg.model.max_seq_len), dtype=tf.int32),
            "attention_mask": tf.ones((1, cfg.model.max_seq_len), dtype=tf.int32),
            "pixel_values": tf.zeros(
                (1, cfg.model.image_size, cfg.model.image_size, 3), dtype=tf.float32
            ),
        },
        training=False,
    )

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
    export_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = latest_weight_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        model.load_weights(str(latest_ckpt))
        print(f"Resumed from checkpoint: {latest_ckpt}")

    summary_writer = tf.summary.create_file_writer(str(log_dir))

    grad_accum_steps = max(cfg.training.gradient_accumulation_steps, 1)
    train_vars = model.trainable_variables
    gradient_accumulator = [
        tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars
    ]
    pending_steps = 0
    checkpoints_written: List[str] = []
    final_train_loss = 0.0
    final_val_loss = 0.0

    global_step = 0
    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        epoch_loss = tf.keras.metrics.Mean()

        for step, (features, labels) in enumerate(
            tqdm(train_ds, desc="train", leave=False)
        ):
            with tf.GradientTape() as tape:
                logits = model(features, training=True)
                loss = masked_cross_entropy(labels=labels, logits=logits)
                scaled_loss = loss / float(grad_accum_steps)

                if cfg.training.mixed_precision:
                    scaled_loss = optimizer.get_scaled_loss(scaled_loss)

            gradients = tape.gradient(scaled_loss, train_vars)
            if cfg.training.mixed_precision:
                gradients = optimizer.get_unscaled_gradients(gradients)

            for acc, grad in zip(gradient_accumulator, gradients):
                if grad is not None:
                    acc.assign_add(grad)
            pending_steps += 1

            should_apply = ((step + 1) % grad_accum_steps) == 0
            if should_apply:
                optimizer.apply_gradients(
                    [(acc, var) for acc, var in zip(gradient_accumulator, train_vars)]
                )
                for acc in gradient_accumulator:
                    acc.assign(tf.zeros_like(acc))
                pending_steps = 0

            epoch_loss.update_state(loss)
            global_step += 1

            if global_step % 20 == 0:
                with summary_writer.as_default():
                    tf.summary.scalar(
                        "train/loss", epoch_loss.result(), step=global_step
                    )

        if pending_steps > 0:
            optimizer.apply_gradients(
                [(acc, var) for acc, var in zip(gradient_accumulator, train_vars)]
            )
            for acc in gradient_accumulator:
                acc.assign(tf.zeros_like(acc))
            pending_steps = 0

        with summary_writer.as_default():
            tf.summary.scalar("train/epoch_loss", epoch_loss.result(), step=epoch + 1)

        final_train_loss = float(epoch_loss.result())
        print(f"train_loss={final_train_loss:.6f}")

        val_loss = tf.keras.metrics.Mean()
        for features, labels in tqdm(val_ds, desc="val", leave=False):
            logits = model(features, training=False)
            loss = masked_cross_entropy(labels=labels, logits=logits)
            val_loss.update_state(loss)

        with summary_writer.as_default():
            tf.summary.scalar("val/loss", val_loss.result(), step=epoch + 1)
        final_val_loss = float(val_loss.result())
        print(f"val_loss={final_val_loss:.6f}")

        ckpt_path = (
            checkpoint_dir / f"epoch-{epoch + 1:04d}-step-{global_step:08d}.weights.h5"
        )
        model.save_weights(str(ckpt_path))
        checkpoints_written.append(str(ckpt_path))
        print(f"checkpoint={ckpt_path}")

    tf.saved_model.save(model, str(export_dir))
    print(f"Saved model export to {export_dir}")

    registry_path = export_dir.parent.parent / "registry" / "model_registry.v1.jsonl"
    run_row = {
        "ts": now_iso(),
        "run_id": f"tf-qwen3-vl-text-{uuid.uuid4().hex[:12]}",
        "kind": "training_run",
        "track": "tf_qwen3_vl_text",
        "constitution_id": constitution_id,
        "manifest_path": str(manifest_path),
        "config_path": str(config_path.resolve()),
        "dataset_checksums": checksum_summary,
        "training_examples": {
            "path": training_examples_summary["path"],
            "total": training_examples_summary["total"],
        },
        "eval": {
            "required": True,
            "val_loss": final_val_loss,
        },
        "metrics": {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss,
        },
        "rollback_checkpoint": str(latest_ckpt) if latest_ckpt else None,
        "checkpoints_written": checkpoints_written,
        "export_dir": str(export_dir),
    }
    append_registry_row(registry_path, run_row)
    print(f"registry_row={registry_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TensorFlow Qwen3-VL fine-tuning harness"
    )
    parser.add_argument("--config", type=str, default="/workspace/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = read_config(config_path)
    set_seed(cfg.seed)
    run_training(cfg, config_path=config_path)


if __name__ == "__main__":
    main()
