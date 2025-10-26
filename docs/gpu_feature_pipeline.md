# GPU Feature Extraction Pipeline

## Background
- Training previously relied on `KaldiDiarizationDataset` to compute STFT → log-mel features on **every** `__getitem__` call using `librosa`.  
- Even after increasing `num_workers`/`prefetch_factor`, CPU usage stayed near 100 % while RTX 4090 utilization remained 1–4 %.  
- Bottleneck: per-batch STFT/log-mel happens on CPU workers, so the GPU waits for feature tensors.

## Changes in this branch
1. **Config knob** (`feature_stage: dataset|cuda`, new CLI arg `--feature-stage`). Defaults to `dataset` for backward compatibility; `examples/train.yaml` sets `cuda` to opt in.  
2. **Dataset output split**: when stage=`cuda`, `KaldiDiarizationDataset` now returns raw waveform chunks + diarization spans instead of features.  
3. **Prefetcher upgrade**: `CUDAPrefetcher` transfers either feature tensors (legacy) or padded waveforms/metadata to GPU streams.  
4. **On-GPU feature builder** (`build_cuda_batch`): uses `torch.stft` + `torchaudio.functional.melscale_fbanks`, applies normalization, context splicing, subsampling, and speaker selection on CUDA.  
5. **Model compatibility**: pad/label utilities now respect tensor devices/dtypes to avoid CPU fallbacks.

## Expected impact
- STFT/log-mel moves from CPU workers to CUDA cores, so GPU utilization should rise substantially.  
- CPU threads mainly handle disk I/O + span extraction, reducing the 50 % `c=` share observed in progress bars.  
- Prefetch pipeline ensures tensors are still ready on the training stream, so no changes to the optimizer or metrics code.

## Validation plan
1. Run a short epoch with `feature_stage=cuda` and compare batch shapes/DER against the legacy path to ensure parity.  
2. Use the existing `c/f/b` timing logs plus `nvidia-smi dmon` to confirm GPU time increases while CPU drops.  
3. Optionally add a regression test that feeds a fixed waveform through both pipelines and checks that log-mel tensors match within tolerance.

## Next steps
- Enable AMP (`torch.cuda.amp`) to further boost throughput once GPU utilization is higher.  
- Consider caching diarization spans or moving audio decoding to GPU-compatible formats if storage bandwidth becomes the new bottleneck.
