#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.profiler import profile, ProfilerActivity

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "eend") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "eend"))

from eend.backend.models import get_model


def load_config(cfg_path: Path) -> SimpleNamespace:
    cfg = yaml.safe_load(cfg_path.read_text())
    ns = SimpleNamespace(**cfg)
    ns.device = torch.device("cpu")
    ns.detach_attractor_loss = bool(cfg.get("detach_attractor_loss", False))
    ns.vad_loss_weight = float(cfg.get("vad_loss_weight", 0.0))
    return ns


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model operations.")
    parser.add_argument("--config", type=Path, default=Path("examples/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("experiment/profile_report.txt"))
    args = parser.parse_args()

    cfg_ns = load_config(args.config)
    model = get_model(cfg_ns)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    feat_dim = cfg_ns.feature_dim * (1 + 2 * cfg_ns.context_size)
    length = max(cfg_ns.num_frames, 32)
    dummy_x = torch.randn(1, length, feat_dim)
    dummy_ts = torch.zeros(1, length, cfg_ns.num_speakers)
    dummy_nspk = torch.full((1,), cfg_ns.num_speakers, dtype=torch.int64)

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with torch.no_grad():
            model(dummy_x, dummy_ts, dummy_nspk, cfg_ns)

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(table)
    print(table)


if __name__ == "__main__":
    main()
