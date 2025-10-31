#!/usr/bin/env python3
"""Compare float32 and INT8 checkpoints produced by infer/export utilities."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "eend"):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from eend.backend.models import get_model  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare float32 vs INT8 checkpoints.")
    parser.add_argument("--float-ckpt", type=Path, required=True,
                        help="Path to float32 snapshot (e.g., float_snapshot_epochs_XX.pt)")
    parser.add_argument("--int8-ckpt", type=Path, required=True,
                        help="Path to quantized checkpoint (quantized_epochs_XX.pt)")
    parser.add_argument("--config", type=Path, default=Path("examples/train.yaml"),
                        help="Training config YAML used to instantiate the model")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write comparison results as JSON")
    parser.add_argument("--verbose", action="store_true",
                        help="Print dtype distribution details")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping")

    ns = SimpleNamespace(**data)
    setattr(ns, "device", torch.device("cpu"))
    setattr(ns, "detach_attractor_loss", bool(data.get("detach_attractor_loss", False)))
    setattr(ns, "vad_loss_weight", float(data.get("vad_loss_weight", 0.0)))
    return data, ns


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise KeyError(f"Checkpoint {path} missing 'model_state'")
    return payload


def _model_stats(model: torch.nn.Module) -> Tuple[int, int, Counter]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtype_counter = Counter(p.dtype for p in model.parameters())
    return total, trainable, dtype_counter


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def main() -> None:
    args = _parse_args()

    cfg_dict, cfg_ns = _load_yaml(args.config)

    base_model = get_model(cfg_ns)

    float_payload = _load_checkpoint(args.float_ckpt)
    int8_payload = _load_checkpoint(args.int8_ckpt)

    float_model = copy.deepcopy(base_model)
    float_model.load_state_dict(float_payload["model_state"])

    int8_model = torch.quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    int8_model.load_state_dict(int8_payload["model_state"])

    float_total, float_trainable, float_dtypes = _model_stats(float_model)
    int8_total, int8_trainable, int8_dtypes = _model_stats(int8_model)

    result = {
        "float": {
            "path": str(args.float_ckpt),
            "file_size_mb": round(_size_mb(args.float_ckpt), 3),
            "total_params": float_total,
            "trainable_params": float_trainable,
            "dtype_count": {str(k): v for k, v in float_dtypes.items()},
            "metadata": float_payload.get("metadata", {}),
        },
        "int8": {
            "path": str(args.int8_ckpt),
            "file_size_mb": round(_size_mb(args.int8_ckpt), 3),
            "total_params": int8_total,
            "trainable_params": int8_trainable,
            "dtype_count": {str(k): v for k, v in int8_dtypes.items()},
            "metadata": int8_payload.get("metadata", {}),
        },
    }

    print("=== Checkpoint Comparison ===")
    for key, info in result.items():
        print(f"[{key.upper()}]")
        print(f" path            : {info['path']}")
        print(f" size (MB)       : {info['file_size_mb']}")
        print(f" total params    : {info['total_params']}")
        print(f" trainable params: {info['trainable_params']}")
        if args.verbose:
            print(f" dtypes          : {info['dtype_count']}")
        meta = info.get("metadata") or {}
        if meta:
            print(f" metadata        : {json.dumps(meta, indent=2) if args.verbose else meta}")
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved JSON report to {args.json}")


if __name__ == "__main__":
    main()
