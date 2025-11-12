#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "eend") not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT / "eend"))

from torchinfo import summary
from eend.backend.models import get_model


class ModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, args: SimpleNamespace):
        super().__init__()
        self.model = model
        self.args = args

    def forward(self, xs: torch.Tensor, ts: torch.Tensor, nspk: torch.Tensor) -> torch.Tensor:
        return self.model(xs, ts, nspk, self.args)


def load_config(cfg_path: Path) -> SimpleNamespace:
    cfg = yaml.safe_load(cfg_path.read_text())
    ns = SimpleNamespace(**cfg)
    ns.device = torch.device("cpu")
    ns.detach_attractor_loss = bool(cfg.get("detach_attractor_loss", False))
    ns.vad_loss_weight = float(cfg.get("vad_loss_weight", 0.0))
    return ns


def print_summary(model: torch.nn.Module, args: SimpleNamespace) -> None:
    feat_dim = args.feature_dim * (1 + 2 * args.context_size)
    length = max(args.num_frames, 32)
    dummy_x = torch.randn(1, length, feat_dim)
    dummy_ts = torch.zeros(1, length, args.num_speakers)
    dummy_nspk = torch.full((1,), args.num_speakers, dtype=torch.int64)
    wrapped = ModelWrapper(model, args)
    summary(
        wrapped,
        input_data=(dummy_x, dummy_ts, dummy_nspk),
        col_names=("kernel_size", "num_params"),
        depth=4,
        verbose=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure linear parameter ratio.")
    parser.add_argument("--config", type=Path, default=Path("examples/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    cfg_ns = load_config(args.config)
    model = get_model(cfg_ns)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    linear_params = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            linear_params += sum(p.numel() for p in module.parameters())

    ratio = linear_params / total_params if total_params else 0.0
    print(f"Total params : {total_params:,}")
    print(f"Linear params: {linear_params:,}")
    print(f"Ratio        : {ratio * 100:.2f}%")
    print_summary(model, cfg_ns)


if __name__ == "__main__":
    main()
