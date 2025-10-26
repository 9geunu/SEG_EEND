# CUDA Feature Pipeline Rollout Report

## 1. 초기 상황 진단
- 학습 속도: 배치당 ~1.3–1.4초, epoch 당 약 5시간 (14,239 배치 기준).
- 리소스: GPU 활용률 2–4%, CPU 95% 이상 포화, 시스템 RAM 7 GiB 사용.
- 원인: `KaldiDiarizationDataset`이 모든 STFT/로그멜/스플라이싱을 CPU 워커에서 수행하여 GPU가 대기.

## 2. 개선 계획
1. **DataLoader 병렬화**: `num_workers`, `prefetch_factor`, `persistent_workers` 인자화 및 기본값 상향, `num_frames` 확대.
2. **특징 생성 캐시**: `librosa.filters.mel` 캐시 도입으로 워커 warm-up 비용 축소.
3. **GPU 파이프라인**: `feature_stage` 옵션 추가, Dataset/Prefetcher/`build_cuda_batch` 개편으로 STFT~로그멜을 CUDA에서 실행.
4. **문서화 & 검증**: 계획 문서(`gpu_feature_plan.md`, `gpu_feature_pipeline.md`) 작성, torchaudio API 호환성·pad 로직 검증.

## 3. 시행 현황
- DataLoader: 32개의 worker와 prefetch_factor 6, persistent workers 활성화, 배치 시퀀스 길이 800으로 증가.
- 캐싱: mel 필터를 `(sr, n_fft, n_mels, dtype)` 키로 캐시.
- GPU 파이프라인: wave chunk + span 메타데이터를 GPU로 전송 후 `torch.stft` + `torchaudio` mel 필터, normalization, splicing, subsampling, speaker selection 단계 구현. pad 함수는 디바이스/ dtype을 유지하도록 수정.
- 호환성: torchaudio의 `melscale_fbanks` 시그니처 차이(n_stft, dtype, f_min/f_max)와 matmul 축 문제를 해결하고, 명확한 에러 메시지 추가.

## 4. 성과
- 배치 시간 `bt≈0.31s`, 1초당 2.5 iteration. Epoch(5,282 배치) ≈ 27분 → 100 epoch ≈ 45시간.
- 초기 대비 학습 시간 11배 단축(510시간 → 45시간).
- GPU Util 40%, CPU Load 8–10%로 병목 이동. 시스템 RAM 30 GiB 사용(파형 버퍼 때문).

## 5. 결론 및 다음 단계
- **결론**: CPU 병목 제거 및 GPU 기반 파이프라인 안정화로 학습 효율이 크게 향상되었으며, 향후 실험에 집중할 수 있는 기반 마련.
- **다음 단계**:
  1. AMP(FP16) 적용으로 GPU 활용률 추가 개선.
  2. 워커/프리패치 조정이나 chunk 길이 최적화로 RAM 사용량 관리.
  3. CPU vs GPU 파이프라인 출력 비교 테스트 자동화.
  4. wave-level augmentation 등 GPU에서의 추가 연산 실험.
