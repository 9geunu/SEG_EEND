#!/usr/bin/env python3
"""Export a dynamically quantized INT8 checkpoint from the latest baseline model.

Usage
-----
python tools/export_int8.py \
    --config examples/train.yaml \
    --checkpoint-dir experiment/baseline/models \
    --output experiment/baseline/models/checkpoint_latest_int8.pt

If --checkpoint / --checkpoint-dir / --output are omitted the script will infer
paths from the training config and drop the INT8 file next to the source
checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import yaml

from types import SimpleNamespace

from eend.backend.models import get_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert trained model to dynamic INT8.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/train.yaml"),
        help="YAML config used for baseline training (default: examples/train.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Specific checkpoint path (.tar). Overrides --checkpoint-dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing checkpoint_*.tar (defaults to config.output_path/models).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for INT8 checkpoint (defaults to <stem>_int8.pt next to source).",
    )
    return parser.parse_args()


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} is not a mapping.")
    return data


def _simple_namespace_from_config(cfg: Dict[str, Any]) -> SimpleNamespace:
    required_fields: Iterable[str] = (
        "model_type",
        "feature_dim",
        "context_size",
        "hidden_size",
        "encoder_units",
        "transformer_encoder_n_heads",
        "transformer_encoder_n_layers",
        "transformer_encoder_dropout",
        "attractor_loss_ratio",
        "attractor_encoder_dropout",
        "attractor_decoder_dropout",
    )

    missing = [key for key in required_fields if key not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    namespace = SimpleNamespace(**{key: cfg[key] for key in cfg})
    # Defaults used by get_model / downstream helper code
    setattr(namespace, "device", torch.device("cpu"))
    # Guard optional keys consumed by model constructor
    setattr(namespace, "detach_attractor_loss", bool(cfg.get("detach_attractor_loss", False)))
    setattr(namespace, "vad_loss_weight", float(cfg.get("vad_loss_weight", 0.0)))
    return namespace


def _latest_checkpoint(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    candidates = sorted(ckpt_dir.glob("checkpoint_*.tar"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.tar files under {ckpt_dir}")
    return candidates[-1]


def _align_state_dict(source: Dict[str, torch.Tensor], target_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    has_module_prefix = any(k.startswith("module.") for k in source)
    target_expects_module = any(str(k).startswith("module.") for k in target_keys)
    if has_module_prefix and not target_expects_module:
        return {k.replace("module.", "", 1): v for k, v in source.items()}
    if not has_module_prefix and target_expects_module:
        return {f"module.{k}": v for k, v in source.items()}
    return source


def _determine_output_path(output: Optional[Path], checkpoint_path: Path) -> Path:
    if output is not None:
        return output
    suffix = checkpoint_path.suffix
    stem = checkpoint_path.stem
    return checkpoint_path.with_name(f"{stem}_int8.pt" if suffix else f"{stem}_int8")


def export_dynamic_int8() -> None:
    args = _parse_args()

    cfg = _load_yaml_config(args.config)

    checkpoint_path: Optional[Path] = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = args.checkpoint_dir
        if ckpt_dir is None:
            output_path = cfg.get("output_path")
            if not output_path:
                raise ValueError("Config must define 'output_path' when --checkpoint is not provided.")
            ckpt_dir = Path(output_path).expanduser().resolve() / "models"
        else:
            ckpt_dir = ckpt_dir.expanduser().resolve()
        checkpoint_path = _latest_checkpoint(ckpt_dir)
    checkpoint_path = checkpoint_path.expanduser().resolve()

    ckpt_payload = torch.load(checkpoint_path, map_location="cpu")

    model_state: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(ckpt_payload, dict):
        model_state = ckpt_payload.get("model_state") or ckpt_payload.get("model_state_dict")
    if model_state is None:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'model_state' or 'model_state_dict'.")

    args_ns = _simple_namespace_from_config(cfg)

    model = get_model(args_ns)
    model.eval()
    aligned_state = _align_state_dict(model_state, model.state_dict().keys())
    model.load_state_dict(aligned_state)

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.eval()

    export_path = _determine_output_path(args.output, checkpoint_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = {
        "epoch": int(ckpt_payload.get("epoch", -1)) if isinstance(ckpt_payload, dict) else -1,
        "source_checkpoint": str(checkpoint_path),
        "config": cfg,
        "quantization": {
            "method": "dynamic",
            "dtype": "torch.qint8",
            "modules": ["torch.nn.Linear"],
        },
    }

    torch.save({
        "model_state": quantized_model.state_dict(),
        "metadata": meta,
    }, export_path)

    # Provide a lightweight manifest for scripting environments.
    manifest_path = export_path.with_suffix(export_path.suffix + ".json")
    manifest_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✔ Saved INT8 checkpoint to {export_path}")
    print(f"ℹ Metadata written to {manifest_path}")


if __name__ == "__main__":
    export_dynamic_int8()
