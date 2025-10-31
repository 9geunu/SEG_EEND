# INT8 동적 양자화 계획 및 현황

## 1. 도입 배경
- **목표**: 베이스라인 SEG_EEND 모델을 모바일 기기에서 실행할 수 있도록 메모리 사용량과 추론 시간을 줄이면서도 DER 저하를 최소화한다.
- **선택 이유**: LoRA는 학습 효율 개선에는 도움이 되지만 추론 파라미터 수나 FLOP을 줄이지 못하므로, 학습 완료 후 `torch.quantization.quantize_dynamic`을 이용한 동적 INT8 양자화를 우선 적용하기로 결정했다.
- **검증 전략**: 학습이 끝난 checkpoint에 바로 적용해 추가 파인튜닝 없이 INT8 모델을 얻고, dev/test 세트에서 DER을 재측정해 성능 하락 여부를 평가한다.

## 2. 구현된 코드 구성요소
### 2.1 `tools/export_int8.py`
- 최신 `checkpoint_*.tar`를 자동으로 탐색해 로드하고, `torch.quantization.quantize_dynamic(..., {torch.nn.Linear}, dtype=torch.qint8)`을 호출해 INT8 state dict를 생성한다.
- 레포 구조를 고려해 `sys.path`에 프로젝트 루트와 `eend/` 디렉터리를 삽입하여 독립 실행 환경에서도 `eend.backend.models`를 import할 수 있도록 했다.
- 변환 결과는 원본 체크포인트와 같은 디렉터리에 `<원본>_int8.pt` 형태로 저장하며, JSON 메타데이터(`epochs`, `source_checkpoint`, `quantization` 정보)를 함께 기록한다.
- CLI 인자: `--config`, `--checkpoint`(선택), `--checkpoint-dir`, `--output`. 지정하지 않으면 `config.output_path/models`에서 최신 checkpoint를 사용한다.

### 2.2 `eend/infer.py`
- `--quantize-dynamic` 플래그 추가. 활성화하면 평균된 float 체크포인트에 대해 CPU 모드에서 동적 INT8 양자화를 수행한다.
- 양자화가 끝나면 추론을 그대로 진행하면서, 결과 모델을 `experiment/quantized/models/quantized_epochs_<epochs>.pt`에 저장하고 동일한 메타데이터를 포함시킨다. 디렉터리가 없으면 자동 생성한다.
- INT8 변환 시 GPU를 사용할 수 없으므로, `args.device.type != "cpu"`이면 예외를 발생시켜 CPU 추론만 허용한다.

## 3. `dataloader-optimizations` 브랜치 변경점 요약
- `examples/train.yaml`
  - `feature_stage: cuda`를 추가해 로그멜 추출을 DataLoader 이후 GPU 파이프라인에서 처리하도록 설정.
  - `num_frames`를 500 → 800으로 늘려 긴 chunk를 한 번에 학습해 GPU 활용도를 높임.
  - `prefetch_factor`, `persistent_workers` 등을 조정해 데이터 공급 병목을 완화.
- `docs/gpu_feature_plan.md`에 기재된 대로 GPU feature 추출 로직을 준비해 CPU와 동일한 로그멜 특성을 유지하도록 설계했다. 모델 구조, 입력 텐서 차원, checkpoint 형식은 기존과 동일하다.

## 4. 추론 호환성 평가
- `examples/infer.yaml`은 `context_size`, `feature_dim`, `frame_size`, `frame_shift`, `hidden_size`, `encoder_units`, `transformer` 관련 하이퍼파라미터를 학습 설정과 동일하게 유지하고 있어 구조적으로 호환된다.
- `models_path`와 checkpoint 명명 규칙이 변하지 않았으므로 `average_checkpoints` 단계에서 문제가 없다. 단, 실제 저장된 epoch 번호가 `epochs: 90-100` 범위에 존재하는지 확인이 필요하다.
- 학습 파이프라인이 GPU에서 로그멜을 생성하더라도 최종 입력은 기존과 동일한 공간이므로, 추론 스크립트가 CPU에서 특성 계산을 수행해도 가중치 적용에는 영향이 없다. 다만 dev/test에서 DER을 재측정해 GPU/CPU 전처리 간 차이가 없는지 확인 권장.
- INT8 추론 시에는 `--quantize-dynamic`, `gpu: 0` 조합으로 실행하고, 생성된 RTTM과 float 모델 결과를 비교해 DER 변화를 모니터링한다.

## 5. 향후 작업 제안
1. `examples/infer.yaml`을 사용해 float vs. INT8 모델의 DER을 동일 데이터셋에서 측정하고, 허용 가능한 성능 범위(예: +0.5% 이내)인지 문서화한다.
2. 필요하면 INT8 모델을 모바일 러너(ONNX Runtime, PyTorch Mobile 등)에 탑재해 실제 디바이스에서 지연 시간을 측정한다.
3. GPU feature 추출 경로와 CPU 경로의 수치 차이를 검증하는 회귀 테스트를 추가해 지속적으로 동등성을 확인한다.
