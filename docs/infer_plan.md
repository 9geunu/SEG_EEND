# Inference Parity & INT8 Evaluation Plan

## 목표
- 학습에서 사용한 GPU 로그멜 파이프라인(STFT → mel 필터 → 로그/정규화 → 컨텍스트 확장 → 서브샘플링/패딩)을 추론에서도 동일하게 재현한다.
- CPU 환경(float/INT8 공통)에서 동일 파이프라인을 실행하도록 함으로써 분포 불일치를 제거하고, 양자화 영향만 분리해 평가한다.

## 진행 단계
1. **학습 전처리 분석**
   - `train.py`의 `build_cuda_batch` 흐름을 읽어 STFT 파라미터(center=True, pad_mode="reflect"), mel filter(`torchaudio.functional.melscale_fbanks`, Slaney norm), `log10`, mean/var normalization, context splicing, subsampling, speaker 선택/패딩 방식 등을 정리.

2. **공용 PyTorch 구현으로 리팩터링**
   - 동일한 연산을 `compute_torch_logmel`로 추출(`eend/common_utils/torch_features.py`).
   - `train.py`와 평가 스크립트가 이 함수를 재사용하도록 수정하여 CPU/GPU 어느 장치에서도 같은 코드를 호출하도록 통일.

3. **회귀 테스트**
   - `tools/feature_parity_check.py`를 리팩터링하여 동일 waveform을 CPU/GPU 경로에 통과시킨 뒤 RMSE, MAX, SNR, allclose 여부를 출력.
   - 실행 결과: 평균 RMSE ≈ 3.5e-07, SNR ≈ 130 dB, `features_allclose=True`로 확인되어 CPU/GPU 로그멜이 거의 완벽히 일치함을 확인.

4. **추론 파이프라인 정렬**
   - `infer.py`에서 `compute_torch_logmel`을 사용하도록 업데이트해 CPU 기반 추론(float/INT8)의 입력 분포를 학습 시 구성과 동일하게 맞춤.
   - `--feature-stage` 옵션을 추가해 필요 시 기존 CPU feature 경로(`dataset`)로도 fallback 가능.

5. **양자화 평가 (진행 예정)**
   - 전처리 동등성을 확보한 뒤 float vs. INT8 모델을 CPU에서 추론해 DER을 비교하고, 양자화 영향만 측정한다.
   - 이후 필요 시 GPU 추론 옵션(`--feature-stage cuda`)을 사용해 성능 검증 및 문서화를 진행한다.

## 남은 작업
- CPU 기반 float/INT8 DER 측정 및 결과 기록.
- 필요 시 문서/예제 명령 업데이트.
