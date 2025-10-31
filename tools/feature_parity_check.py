#!/usr/bin/env python3
"""Compare CPU vs GPU log-mel feature pipelines on fixed waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "eend"):
    path_str = str(candidate)
    if candidate.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from eend.common_utils.diarization_dataset import KaldiDiarizationDataset
from eend.train import build_cuda_batch


def _load_config(path: Path) -> Dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} does not contain a mapping")
    return cfg


def _build_args(cfg: Dict) -> SimpleNamespace:
    required = [
        "frame_size",
        "frame_shift",
        "sampling_rate",
        "feature_dim",
        "context_size",
        "subsampling",
        "num_speakers",
        "num_frames",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in config: {', '.join(missing)}")

    return SimpleNamespace(
        frame_size=int(cfg["frame_size"]),
        frame_shift=int(cfg["frame_shift"]),
        sampling_rate=int(cfg["sampling_rate"]),
        feature_dim=int(cfg["feature_dim"]),
        context_size=int(cfg.get("context_size", 0)),
        subsampling=int(cfg.get("subsampling", 1)),
        num_speakers=int(cfg.get("num_speakers", 0)),
        num_frames=int(cfg.get("num_frames", 0)),
        input_transform=str(cfg.get("input_transform", "logmel")),
    )


def _dataset_kwargs(cfg: Dict, data_dir: str) -> Dict:
    return dict(
        data_dir=data_dir,
        chunk_size=int(cfg.get("num_frames", 0)),
        context_size=int(cfg.get("context_size", 0)),
        feature_dim=int(cfg["feature_dim"]),
        frame_shift=int(cfg["frame_shift"]),
        frame_size=int(cfg["frame_size"]),
        input_transform=str(cfg.get("input_transform", "logmel")),
        n_speakers=int(cfg.get("num_speakers", 0)),
        sampling_rate=int(cfg["sampling_rate"]),
        shuffle=False,
        subsampling=int(cfg.get("subsampling", 1)),
        use_last_samples=bool(cfg.get("use_last_samples", True)),
        min_length=int(cfg.get("min_length", 0)),
    )


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a - b
    return float(torch.sqrt(torch.mean(diff * diff)).item())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def _snr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    noise_power = torch.mean(noise * noise)
    signal_power = torch.mean(signal * signal)
    if noise_power == 0:
        return float("inf")
    return float(10.0 * torch.log10(signal_power / noise_power).item())


def compare_samples(
    cpu_features: torch.Tensor,
    cpu_labels: torch.Tensor,
    gpu_features: torch.Tensor,
    gpu_labels: torch.Tensor,
) -> Dict[str, float]:
    gpu_features = gpu_features.to(cpu_features.device)
    gpu_labels = gpu_labels.to(cpu_labels.device)

    feature_rmse = _rmse(cpu_features, gpu_features)
    feature_max = _max_abs(cpu_features, gpu_features)
    label_rmse = _rmse(cpu_labels, gpu_labels)
    label_max = _max_abs(cpu_labels, gpu_labels)

    noise = gpu_features - cpu_features
    feature_snr = _snr(cpu_features, noise)

    allclose = bool(torch.allclose(cpu_features, gpu_features, atol=1e-3, rtol=1e-3))
    labels_close = bool(torch.allclose(cpu_labels, gpu_labels, atol=1e-3, rtol=1e-3))

    return {
        "feature_rmse": feature_rmse,
        "feature_max": feature_max,
        "feature_snr_db": feature_snr,
        "label_rmse": label_rmse,
        "label_max": label_max,
        "features_allclose": allclose,
        "labels_allclose": labels_close,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU vs CPU feature parity check")
    parser.add_argument("--config", type=Path, default=Path("examples/train.yaml"))
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this comparison")

    cfg = _load_config(args.config)
    data_dir = args.data_dir or cfg.get("valid_data_dir") or cfg.get("train_data_dir")
    if not data_dir:
        raise ValueError("Data directory must be provided via --data-dir or config valid/train path")

    dataset_kwargs = _dataset_kwargs(cfg, data_dir)
    cpu_dataset = KaldiDiarizationDataset(feature_stage="dataset", **dataset_kwargs)
    gpu_dataset = KaldiDiarizationDataset(feature_stage="cuda", **dataset_kwargs)

    if len(cpu_dataset) == 0:
        raise RuntimeError(f"Dataset at {data_dir} is empty")

    limit = min(args.samples, len(cpu_dataset))
    indices = list(range(limit))

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    compare_args = _build_args(cfg)

    stats: List[Dict[str, float]] = []

    for idx in indices:
        cpu_feat, cpu_lbl, rec = cpu_dataset[idx]
        wave_sample = gpu_dataset[idx]

        batch = {
            "audio": wave_sample["waveform"].unsqueeze(0).to(device),
            "lengths": torch.tensor([wave_sample["waveform"].shape[0]], device=device),
            "spans": [wave_sample["spans"]],
            "speaker_count": [wave_sample["speaker_count"]],
            "names": [wave_sample["names"]],
        }

        gpu_feat, gpu_lbl, _ = build_cuda_batch(batch, compare_args)
        gpu_feat = gpu_feat.squeeze(0).cpu()
        gpu_lbl = gpu_lbl.squeeze(0).cpu()

        cpu_feat = cpu_feat.float()
        cpu_lbl = cpu_lbl.float()

        sample_stats = compare_samples(cpu_feat, cpu_lbl, gpu_feat, gpu_lbl)
        sample_stats["record"] = rec
        stats.append(sample_stats)

        print(json.dumps({"record": rec, **sample_stats}, ensure_ascii=False))

    mean_metrics = {
        "feature_rmse": sum(s["feature_rmse"] for s in stats) / len(stats),
        "feature_max": max(s["feature_max"] for s in stats),
        "label_rmse": sum(s["label_rmse"] for s in stats) / len(stats),
        "label_max": max(s["label_max"] for s in stats),
        "feature_snr_db": sum(s["feature_snr_db"] for s in stats) / len(stats),
        "features_allclose": all(s["features_allclose"] for s in stats),
        "labels_allclose": all(s["labels_allclose"] for s in stats),
        "samples": len(stats),
    }

    summary = json.dumps(mean_metrics, ensure_ascii=False)
    print(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(summary + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
