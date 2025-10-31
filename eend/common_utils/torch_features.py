#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import torch
import torchaudio.functional as AF
from typing import List, Tuple

from backend.models import pad_labels, pad_sequence  # reuse existing helpers


def _fft_size(frame_size: int) -> int:
    return 1 << (frame_size - 1).bit_length()


def _valid_frame_count(num_samples: int, fft_size: int, frame_shift: int) -> int:
    if num_samples <= 0:
        return 0
    pad = fft_size // 2
    n = num_samples + 2 * pad - fft_size
    if n < 0:
        return 0
    frames = n // frame_shift + 1
    if num_samples % frame_shift == 0:
        frames = max(frames - 1, 0)
    return frames


def _splice_tensor(feat: torch.Tensor, context: int) -> torch.Tensor:
    if context <= 0:
        return feat
    padded = torch.nn.functional.pad(feat, (0, 0, context, context))
    unfolded = padded.unfold(0, 2 * context + 1, 1)
    return unfolded.reshape(feat.shape[0], -1)


def _subsample_tensor(t: torch.Tensor, subsampling: int) -> torch.Tensor:
    if subsampling <= 1:
        return t
    return t[::subsampling]


def _select_top_speakers(labels: torch.Tensor, max_speakers: int) -> torch.Tensor:
    if max_speakers and labels.shape[1] > max_speakers:
        totals = labels.sum(dim=0)
        topk = torch.topk(totals, k=max_speakers).indices
        topk, _ = torch.sort(topk)
        labels = labels[:, topk]
    return labels


def compute_torch_logmel(
    audio: torch.Tensor,
    lengths: torch.Tensor,
    spans_batch,
    args,
    device: torch.device | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    device = device if device is not None else audio.device
    audio = audio.to(device)
    lengths = lengths.to(device)

    max_output_speakers = getattr(args, "num_speakers", 0)
    if max_output_speakers is None:
        max_output_speakers = 0

    fft_size = _fft_size(int(args.frame_size))
    window = torch.hann_window(int(args.frame_size), device=device, dtype=audio.dtype)
    stft = torch.stft(
        audio,
        n_fft=fft_size,
        hop_length=int(args.frame_shift),
        win_length=int(args.frame_size),
        window=window,
        return_complex=True,
        center=True,
        pad_mode="reflect",
    )
    spec = stft.abs().pow(2.0).transpose(1, 2)

    mel_fbanks = AF.melscale_fbanks(
        n_freqs=fft_size // 2 + 1,
        f_min=0.0,
        f_max=float(args.sampling_rate) / 2.0,
        n_mels=int(args.feature_dim),
        sample_rate=int(args.sampling_rate),
        norm="slaney",
    ).to(device=device, dtype=spec.dtype)
    if mel_fbanks.shape[0] != spec.shape[2]:
        raise RuntimeError(
            f"Mel filter mismatch: spec_dim={spec.shape[2]} vs fbanks={mel_fbanks.shape[0]}"
        )
    mel = torch.matmul(spec, mel_fbanks)
    mel = torch.log10(torch.clamp(mel, min=1e-10))

    transform = getattr(args, "input_transform", "logmel")
    if transform == "logmel_meannorm":
        mel = mel - mel.mean(dim=1, keepdim=True)
    elif transform == "logmel_meanvarnorm":
        mel = mel - mel.mean(dim=1, keepdim=True)
        std = torch.clamp(mel.std(dim=1, keepdim=True), min=1e-5)
        mel = mel / std

    feature_list: List[torch.Tensor] = []
    label_list: List[torch.Tensor] = []
    speaker_counts: List[int] = []

    for idx in range(audio.size(0)):
        num_frames = min(
            mel.shape[1],
            _valid_frame_count(int(lengths[idx].item()), fft_size, int(args.frame_shift)),
        )
        num_frames = max(num_frames, 1)
        feat = mel[idx, :num_frames, :]

        max_spk = max([span[0] for span in spans_batch[idx]], default=-1) + 1
        max_spk = max(max_spk, 1)
        labels = torch.zeros((num_frames, max_spk), device=device, dtype=feat.dtype)
        for speaker_index, start_f, end_f in spans_batch[idx]:
            start = max(0, min(start_f, num_frames))
            end = max(0, min(end_f, num_frames))
            if end > start and speaker_index < max_spk:
                labels[start:end, speaker_index] = 1.0

        feat = _splice_tensor(feat, int(args.context_size))
        feat = _subsample_tensor(feat, int(args.subsampling))
        labels = _subsample_tensor(labels, int(args.subsampling))
        labels = _select_top_speakers(labels, int(max_output_speakers))

        feature_list.append(feat)
        label_list.append(labels)
        speaker_counts.append(labels.shape[1])

    target_frames = getattr(args, "num_frames", 0)
    if target_frames is None:
        target_frames = 0
    target_frames = int(target_frames)

    if target_frames > 0:
        feature_list, label_list = pad_sequence(feature_list, label_list, target_frames)

    max_speakers = max(speaker_counts) if speaker_counts else 0
    label_list = pad_labels(label_list, max_speakers)

    features = torch.stack(feature_list)
    labels = torch.stack(label_list)
    n_speakers = np.asarray(speaker_counts)
    return features.to(device), labels.to(device), n_speakers
