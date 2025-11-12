#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "eend") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "eend"))

from eend.backend.models import get_model
from eend.common_utils.torch_features import compute_torch_logmel
from eend.infer import get_infer_dataloader


def load_config(cfg_path: Path) -> SimpleNamespace:
    cfg = yaml.safe_load(cfg_path.read_text())
    ns = SimpleNamespace(**cfg)
    ns.device = torch.device("cpu")
    defaults = {
        "num_speakers": 2,
        "feature_stage": "cuda",
        "time_shuffle": False,
        "num_workers": 0,
        "infer_data_dir": cfg.get("infer_data_dir"),
        "rttms_dir": str(Path("experiment") / "profile_rttms"),
        "model_type": cfg.get("model_type", "TransformerEDA"),
        "threshold": cfg.get("threshold", 0.5),
        "median_window_length": cfg.get("median_window_length", 11),
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
            cfg.setdefault(key, value)
    if not getattr(ns, "infer_data_dir", None):
        raise ValueError("infer_data_dir must be set in config or defaults")
    return ns


def load_model(state_path: Path, args: SimpleNamespace, use_quantized: bool) -> torch.nn.Module:
    payload = torch.load(state_path, map_location="cpu")
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    is_quantized_ckpt = meta.get("quantization", {}).get("method") == "dynamic"

    model = get_model(args)
    if use_quantized or is_quantized_ckpt:
        if torch.backends.quantized.engine == 'none':
            torch.backends.quantized.engine = 'fbgemm'
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.eval()

    state = payload.get("model_state") if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_batch(args: SimpleNamespace, batch) -> torch.Tensor:
    if args.feature_stage == "cuda":
        waveform = batch['waveforms'][0].unsqueeze(0).float()
        lengths = torch.tensor([waveform.shape[1]], dtype=torch.long)
        spans = [batch['spans'][0]]
        logmel_args = SimpleNamespace(**vars(args))
        if getattr(logmel_args, "num_frames", 0) <= 0:
            logmel_args.num_frames = 0
        features, _, _ = compute_torch_logmel(
            waveform,
            lengths,
            spans,
            logmel_args,
            device=torch.device("cpu"),
        )
        return features
    return torch.stack(batch['xs']).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EEND inference operations.")
    parser.add_argument("--config", type=Path, default=Path("examples/infer.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to .pt checkpoint containing model_state")
    parser.add_argument("--max-batches", type=int, default=5,
                        help="Number of inference batches to profile")
    parser.add_argument("--quantize-dynamic", action='store_true')
    parser.add_argument("--trace", type=Path,
                        help="Optional chrome trace output path")
    parser.add_argument("--report", type=Path, default=Path("experiment/profile_report.txt"))
    args = parser.parse_args()

    cfg_ns = load_config(args.config)
    dataloader = get_infer_dataloader(cfg_ns)
    model = load_model(args.checkpoint, cfg_ns, args.quantize_dynamic)

    activities = [ProfilerActivity.CPU]
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                feats = prepare_batch(cfg_ns, batch)
                _ = model.estimate_sequential(feats, cfg_ns)[0]
                if idx + 1 >= args.max_batches:
                    break

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(table)
    print(table)
    if args.trace:
        prof.export_chrome_trace(str(args.trace))


if __name__ == "__main__":
    main()
