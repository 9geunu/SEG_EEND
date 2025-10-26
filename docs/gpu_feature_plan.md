# GPU Feature Extraction Refactor Plan

## Goals
- Move log-mel extraction from CPU DataLoader workers onto the GPU stream that already serves model training.
- Avoid re-reading audio where possible and keep compatibility with existing Kaldi-style manifests.

## Approach
1. **Dataset Output**  
   - Add a switch (`feature_stage: dataset|prefetch`) so the dataset can either emit ready-made features (current behavior) or emit raw waveform chunks plus segment labels.  
   - When `prefetch`, `KaldiDiarizationDataset` skips `librosa` transforms and only normalizes chunk boundaries + labels.

2. **GPU Extraction Module**  
   - Introduce `gpu_features.py` with functions built on `torch.stft` and `torchaudio.functional.melscale_fbanks`.  
   - Reuse existing chunk metadata (`frame_size`, `frame_shift`, `context_size`, `subsampling`) to reproduce the same feature shapes.  
   - Support AMP by running in the same CUDA stream as the rest of the training loop.

3. **Integration Point**  
   - Extend `CUDAPrefetcher` so that, after transferring tensors to CUDA, it optionally calls the GPU extractor before yielding the batch.  
   - Because the computation happens after `prefetcher.next()` returns, the model always receives tensors in log-mel space regardless of the pipeline.

4. **Performance Safeguards**  
   - Keep a CPU fallback (current path).  
   - Add unit tests comparing CPU vs GPU features on a fixed waveform to guarantee numerical parity within tolerance.  
   - Gate the feature path via config to allow gradual rollout.

5. **Future Optimizations**  
   - Cache FFT plans per device to minimize kernel setup cost.  
   - Explore batched STFT kernels to amortize launch overhead over multiple utterances, matching the `train_batchsize`.

This document tracks implementation progress; once the GPU pipeline lands, we can remove the `feature_stage` flag and keep GPU as the default for CUDA-enabled runs.
