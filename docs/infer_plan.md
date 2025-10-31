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

5. **실험 케이스 정리**
   - **Case A (baseline 설정)**
     - 명령: `python eend/infer.py -c examples/infer.yaml --feature-stage cuda`
     - 평가: `dscore` → DER ≈ 39.5%, JER ≈ 18.6% → 전처리 정렬 전 대비는 크게 개선됐으나 화자 수 추정 미조정으로 일부 샘플 DER이 높음.
   - **Case B (auto speaker qty)**
     - 명령: `python eend/infer.py -c examples/infer.yaml --feature-stage cuda --estimate-spk-qty -1 --estimate-spk-qty-thr 0.5`
     - 평가: `dscore` → DER ≈ 7.48%, JER ≈ 12.1%, B3-F1 ≈ 0.79 → 화자 수를 임계값 기반으로 추정하면서 DER이 크게 하락.
   - 파라미터 튜닝 목표: `estimate_spk_qty_thr`를 dev 기준으로 추가 그리드 탐색(0.3~0.6)하고, 필요 시 `--threshold`·`--median-window-length`도 재조정.

6. **양자화 평가 (진행 예정)**
   - 전처리 동등성을 확보한 뒤 float vs. INT8 모델을 CPU에서 추론해 DER을 비교하고, 양자화 영향만 측정한다.
   - 이후 `--feature-stage cuda`를 유지해 GPU 학습과 동일한 파이프라인에서 INT8 결과를 분석한다.

## 남은 작업
- `estimate_spk_qty_thr`, `threshold`, `median_window_length` 등에 대한 dev 튜닝으로 DER 재최적화.
- CPU 기반 float/INT8 DER 측정 및 결과 기록.
- 필요 시 문서/예제 명령 업데이트 및 INT8 관련 배포 문서화.
