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

- **cuDNN 재활성화**: `torch.backends.cudnn.enabled = True`로 학습 루프가 다시 cuDNN 커널을 사용하도록 수정(결정성 보존을 위해 `deterministic=True`, `benchmark=False` 유지). 이 변경 이후 미니배치 처리 시간이 0.42 s → 0.16 s로 단축되고, GPU 활용률이 40 % → 44–46 %로 상승했다. GPU가 데이터를 더 빨리 소비하면서 호스트 큐에 머무는 배치 수가 줄어 RAM 점유도 21 GiB → 7 GiB 수준까지 자연스럽게 감소.
- **DataLoader 메모리 절감**:
  - `prefetch_factor`를 CLI 값 그대로 사용하게 해 워커당 최대 1개의 배치만 선적하도록 조정. (이전에는 코드가 강제로 ≥2로 고정돼 워커×배치수만큼 파형이 메모리에 상주.)
  - 기본 `num_workers`를 CPU 코어 수의 1/4로 계산하고, 필요 시만 증설하도록 해 과도한 프로세스 스폰을 방지.
  - 새로운 `--multiprocessing-context` 인자로 Linux 기본값을 `fork`로 설정해 Kaldi 메타데이터를 copy-on-write로 공유. 이로써 32 워커 환경에서 30 GiB까지 치솟던 시스템 RAM 사용량이 7 GiB 전후로 안정화.
- **검증 루프 정렬**: `feature_stage="cuda"`일 때 개발(검증) 데이터도 학습 경로와 동일한 GPU 전처리를 거치도록 `prepare_batch_on_cpu` → CUDA 전송 → `build_cuda_batch` 순서를 재사용. 덕분에 train/dev 모두 동일한 파이프라인에서 측정되어 성능 비교가 깔끔해지고, 검증 단계에서의 CPU 작업 대기도 감소했다.
- **현재 고정 하이퍼파라미터 커맨드**:
  
  ```bash
  torchrun \
    --nproc_per_node=1 \
    --max_restarts=0 \
    --standalone \
    --tee 3 \
    eend/train.py -c examples/train.yaml --ddp --noam-k 1 \
    --train-batchsize 32 \
    --accum-steps 2 \
    --prefetch-factor 1 \
    --multiprocessing-context fork
  ```

  위 커맨드를 기준선으로 삼아 baseline 및 경량화 모델을 모두 동일 조건에서 학습 중.
