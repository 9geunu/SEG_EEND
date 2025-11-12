#!/usr/bin/env python3
"""Measure CPU inference metrics for float32 vs INT8 EEND models."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import psutil
import torch
torch.backends.quantized.engine = 'qnnpack'
import yaml

from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT, REPO_ROOT / "eend"):
    cand_str = str(candidate)
    if candidate.exists() and cand_str not in os.sys.path:
        os.sys.path.insert(0, cand_str)

from eend.backend.models import get_model
from eend.common_utils.torch_features import compute_torch_logmel
from eend.infer import get_infer_dataloader


@dataclass
class MetricResult:
    float_value: float
    int8_value: float

    def improvement_pct(self) -> float:
        if self.float_value == 0:
            return 0.0
        return (self.int8_value - self.float_value) / self.float_value * 100.0


def _load_config(cfg_path: Path) -> Tuple[Dict[str, object], SimpleNamespace]:
    cfg_dict = yaml.safe_load(cfg_path.read_text())
    if not isinstance(cfg_dict, dict):
        raise ValueError("YAML config must be a mapping.")
    ns = SimpleNamespace(**cfg_dict)
    ns.device = torch.device("cpu")
    defaults = {
        "num_speakers": 2,
        "feature_stage": "cuda",
        "time_shuffle": False,
        "num_workers": 0,
        "attractor_loss_ratio": 1.0,
        "attractor_encoder_dropout": 0.1,
        "attractor_decoder_dropout": 0.1,
        "estimate_spk_qty": -1,
        "estimate_spk_qty_thr": -1.0,
        "detach_attractor_loss": False,
        "vad_loss_weight": 0.0,
    }
    for key, value in defaults.items():
        if not hasattr(ns, key):
            setattr(ns, key, value)
            cfg_dict.setdefault(key, value)
    return cfg_dict, ns


def _align_state_dict(state: Dict[str, torch.Tensor], target_keys: Iterable[str]) -> Dict[str, torch.Tensor]:
    has_module = any(k.startswith("module.") for k in state)
    target_module = any(str(k).startswith("module.") for k in target_keys)
    if has_module and not target_module:
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    if not has_module and target_module:
        return {f"module.{k}": v for k, v in state.items()}
    return state


def load_model_state(model_path: Path) -> Dict[str, torch.Tensor]:
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    raise KeyError(f"Checkpoint {model_path} does not contain 'model_state'.")


def build_model(cfg_ns: SimpleNamespace, state_dict: Dict[str, torch.Tensor], quantized: bool = False) -> torch.nn.Module:
    model = get_model(cfg_ns)
    if quantized:
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(_align_state_dict(state_dict, model.state_dict().keys()))
    model.eval()
    return model


def prepare_feature_batches(args: SimpleNamespace, max_batches: int = 32) -> List[torch.Tensor]:
    loader = get_infer_dataloader(args)
    batches: List[torch.Tensor] = []
    for batch in loader:
        if args.feature_stage == "cuda":
            waveform = batch['waveforms'][0].unsqueeze(0).float()
            lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
            spans = [batch['spans'][0]]
            local_args = SimpleNamespace(**vars(args))
            if getattr(local_args, "num_frames", 0) <= 0:
                local_args.num_frames = waveform.shape[1]
            features, _, _ = compute_torch_logmel(waveform, lengths, spans, local_args, device=torch.device("cpu"))
            batches.append(features)
        else:
            # dataset mode already returns features
            feats = torch.stack(batch['xs']) if isinstance(batch['xs'], list) else batch['xs']
            batches.append(feats)
        if len(batches) >= max_batches:
            break
    if not batches:
        raise RuntimeError("Failed to prepare any feature batches for measurement.")
    return batches


def _run_inference(model: torch.nn.Module, batch: torch.Tensor, args: SimpleNamespace) -> torch.Tensor:
    with torch.no_grad():
        return model.estimate_sequential(batch, args)[0]


def measure_inference_time(model: torch.nn.Module, sample: torch.Tensor, args: SimpleNamespace, repeats: int = 100) -> Tuple[float, float]:
    torch.set_grad_enabled(False)
    with torch.no_grad():
        for _ in range(min(10, repeats)):
            _run_inference(model, sample, args)
        times: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            _run_inference(model, sample, args)
            end = time.perf_counter()
            times.append((end - start) * 1000)
    return float(mean(times)), float(stdev(times)) if len(times) > 1 else 0.0


def measure_peak_ram(model: torch.nn.Module, sample: torch.Tensor, args: SimpleNamespace, repeats: int = 20) -> float:
    process = psutil.Process(os.getpid())
    peak = 0.0
    torch.set_grad_enabled(False)
    with torch.no_grad():
        for _ in range(repeats):
            before = process.memory_info().rss / (1024 ** 2)
            _run_inference(model, sample, args)
            after = process.memory_info().rss / (1024 ** 2)
            peak = max(peak, after - before)
    return peak


def measure_throughput(model: torch.nn.Module, batches: List[torch.Tensor], args: SimpleNamespace, iterations: int = 100) -> float:
    total_samples = 0
    total_time = 0.0
    torch.set_grad_enabled(False)
    with torch.no_grad():
        idx = 0
        for _ in range(iterations):
            batch = batches[idx]
            idx = (idx + 1) % len(batches)
            batch_size = batch.shape[0]
            start = time.perf_counter()
            _run_inference(model, batch, args)
            end = time.perf_counter()
            total_samples += batch_size
            total_time += (end - start)
    if total_time == 0:
        return 0.0
    return total_samples / total_time


def bytes_to_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare float32 vs INT8 CPU metrics")
    parser.add_argument("--config", type=Path, default=Path("examples/infer.yaml"))
    parser.add_argument("--float-ckpt", type=Path, required=True)
    parser.add_argument("--int8-ckpt", type=Path, required=True)
    parser.add_argument("--time-iters", type=int, default=100)
    parser.add_argument("--throughput-iters", type=int, default=100)
    parser.add_argument("--peak-iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=32, help="number of batches cached for throughput")
    parser.add_argument("--output", type=Path, default=Path("experiment/benchmarks/int8_cpu_metrics.json"))
    parser.add_argument("--float-der", type=float, default=None)
    parser.add_argument("--int8-der", type=float, default=None)
    parser.add_argument("--float-jer", type=float, default=None)
    parser.add_argument("--int8-jer", type=float, default=None)
    parser.add_argument("--float-b3", type=float, default=None)
    parser.add_argument("--int8-b3", type=float, default=None)
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    cfg_dict, cfg_ns = _load_config(args.config)

    batches = prepare_feature_batches(cfg_ns, max_batches=args.batches)
    sample = batches[0]

    float_state = load_model_state(args.float_ckpt)
    int8_state = load_model_state(args.int8_ckpt)

    float_model = build_model(cfg_ns, float_state, quantized=False)
    int8_model = build_model(cfg_ns, int8_state, quantized=True)

    float_time_mean, float_time_std = measure_inference_time(float_model, sample, cfg_ns, repeats=args.time_iters)
    int8_time_mean, int8_time_std = measure_inference_time(int8_model, sample, cfg_ns, repeats=args.time_iters)

    float_peak = measure_peak_ram(float_model, sample, cfg_ns, repeats=args.peak_iters)
    int8_peak = measure_peak_ram(int8_model, sample, cfg_ns, repeats=args.peak_iters)

    float_throughput = measure_throughput(float_model, batches, cfg_ns, iterations=args.throughput_iters)
    int8_throughput = measure_throughput(int8_model, batches, cfg_ns, iterations=args.throughput_iters)

    float_size = bytes_to_mb(args.float_ckpt)
    int8_size = bytes_to_mb(args.int8_ckpt)

    results = {
        "inference_time_ms": {
            "float_mean": float_time_mean,
            "float_std": float_time_std,
            "int8_mean": int8_time_mean,
            "int8_std": int8_time_std,
        },
        "peak_ram_mb": {
            "float": float_peak,
            "int8": int8_peak,
        },
        "throughput_samples_per_sec": {
            "float": float_throughput,
            "int8": int8_throughput,
        },
        "model_size_mb": {
            "float": float_size,
            "int8": int8_size,
        },
        "accuracy": {
            "float": {
                "DER": args.float_der,
                "JER": args.float_jer,
                "B3F1": args.float_b3,
            },
            "int8": {
                "DER": args.int8_der,
                "JER": args.int8_jer,
                "B3F1": args.int8_b3,
            }
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    print("=== CPU Metric Summary ===")
    print(f"Inference time (ms)  : float {float_time_mean:.2f}±{float_time_std:.2f} | int8 {int8_time_mean:.2f}±{int8_time_std:.2f}")
    print(f"Peak RAM (MB)        : float {float_peak:.2f} | int8 {int8_peak:.2f}")
    print(f"Throughput (samples/s): float {float_throughput:.2f} | int8 {int8_throughput:.2f}")
    print(f"Model size (MB)      : float {float_size:.2f} | int8 {int8_size:.2f}")
    if args.float_der is not None and args.int8_der is not None:
        print(f"DER (%)              : float {args.float_der:.2f} | int8 {args.int8_der:.2f}")
    if args.float_jer is not None and args.int8_jer is not None:
        print(f"JER (%)              : float {args.float_jer:.2f} | int8 {args.int8_jer:.2f}")
    if args.float_b3 is not None and args.int8_b3 is not None:
        print(f"B3-F1                : float {args.float_b3:.2f} | int8 {args.int8_b3:.2f}")


if __name__ == "__main__":
    main()
