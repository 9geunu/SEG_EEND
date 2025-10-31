# 모델 경량화 계획 (LoRA / 프루닝 / 양자화)

본 문서는 기존 SEG_EEND 베이스라인 모델을 경량화하기 위한 전략을 정리한다. 각 방법은 현재 코드 베이스에 대한 필수 변경 사항을 포함한다.

## 1. LoRA(Low-Rank Adapter) 적용
- **대상 모듈**: `backend/models.py`의 Transformer encoder 블록(자기 주의 `torch.nn.MultiheadAttention`, FFN 내부 `torch.nn.Linear`).
- **필요 코드 변경**
  1. `common_utils/lora.py`(신규 파일)에서 LoRA 래퍼 `LoRALinear` 구현. `forward` 시 기본 weight + 저랭크 업데이트를 합산하고, `get_lora_parameters()` 등의 헬퍼 제공.
  2. `backend/models.py`에서 encoder 생성 시 `LoRALinear`로 대체하도록 분기 추가. 예: `if args.use_lora: self.q_proj = LoRALinear(nn.Linear(...), rank=args.lora_rank, alpha=args.lora_alpha)`.
  3. Optimizer 초기화( `backend/updater.py`)에서 LoRA 파라미터 전용 파라미터 그룹을 추가하고, 기존 weight는 `requires_grad=False`로 고정.
  4. 체크포인트 저장/로드(`backend/models.py` 또는 `train.py`) 시 LoRA state를 함께 저장하도록 state dict 확장.
  5. YAML에 `use_lora`, `lora_rank`, `lora_alpha` 등 파라미터 추가.
- **워크플로우**: 베이스라인 체크포인트 로드 → LoRA 파라미터만 학습 → 필요시 merge 옵션 제공.
- **유의 사항**: DDP 시 LoRA 모듈이 올바르게 broadcast되도록 `model.module` 경로 처리, AMP와 호환되는 dtype 유지.

```
Transformer Encoder Block
┌──────────────────────────────────────────────────────────┐
│ Input (B, T, D_model)                                    │
│        │                                                 │
│        ├─ LayerNorm ────────────────────────┐            │
│        │                                     │ residual  │
│        ▼                                     ▼           │
│   Multi-Head Self-Attention                  +───────────┤
│   ├─ Q_proj (Linear) + LoRA Adapter ─┐                  │
│   ├─ K_proj (Linear) + LoRA Adapter ─┤                  │
│   ├─ V_proj (Linear) + LoRA Adapter ─┤                  │
│   └─ Out_proj (Linear) + LoRA Adapter┘                  │
│        │                                                 │
│        ├─ Dropout                                        │
│        ▼                                                 │
│   Add & Norm (Residual)                                  │
│        │                                                 │
│        ├─ LayerNorm ────────────────────────┐            │
│        │                                     │ residual  │
│        ▼                                     ▼           │
│   Feed-Forward Network (FFN)                 +───────────┤
│   ├─ Linear1 + LoRA Adapter ─────┐                      │
│   ├─ Activation (ReLU/GELU)      │                      │
│   └─ Linear2 + LoRA Adapter ─────┘                      │
│        │                                                 │
│        ├─ Dropout                                        │
│        ▼                                                 │
│   Add & Norm (Residual)                                  │
│        │                                                 │
│        ▼                                                 │
│   Output (B, T, D_model)                                 │
└──────────────────────────────────────────────────────────┘
```

## 2. 구조적 프루닝 (Structured Pruning)
- **대상 모듈**: `backend/models.py`의 self-attention 헤드, FFN hidden 유닛.
- **필요 코드 변경**
  1. `common_utils/pruning.py` 신설: 중요도 계산 함수(`compute_head_importance`, `compute_channel_importance`)와 마스크 적용 유틸 구현.
  2. 학습 루프(`train.py`)에 warm-up 후 프루닝 이벤트를 삽입. 예: `if epoch == args.prune_epoch: apply_attention_head_pruning(model, prune_ratio=args.prune_heads)`.
  3. Attention 헤드 프루닝: `MultiheadAttention` weight를 head 단위로 reshape → 중요도 기준 상위 head만 유지 → 나머지 weight를 제거하고 `num_heads`/`embed_dim` 업데이트.
  4. FFN 채널 프루닝: `torch.nn.utils.prune.ln_structured` 등을 사용해 마스크 생성 → 실제 레이어를 새 Linear로 재생성하여 불필요 채널 제거.
  5. 프루닝 이후에 옵티마이저와 스케줄러를 재생성해 차원 변화 반영.
  6. YAML에 `use_pruning`, `prune_epoch`, `prune_heads`, `prune_ffn_ratio` 등 인자 추가.
- **워크플로우**: 베이스라인 체크포인트 로드 → warm-up 학습 → 프루닝 적용 → 추가 fine-tuning.
- **유의 사항**: 프루닝 후 state dict와 DDP sync를 위해 `dist.broadcast_parameters` 호출 검토, 저장 시 프루닝 적용 여부 메타 저장.

### `--use_lora` 전용 학습 경로 설계

경량화 실험이 기존 학습 루프와 얽히지 않도록 LoRA 전용 실행 경로를 아래처럼 분리한다.

1. **새 실행 스크립트**: `eend/train_lora.py`
   - `train.py`를 기준으로 복사하되, `parse_arguments()`에서 `--use_lora`를 기본 활성화.
   - LoRA 관련 인자(`lora_rank`, `lora_alpha`, `lora_dropout`)와 LoRA 모델 구성 함수를 불러온다.

2. **모델 초기화**
   - `get_model(args)` 호출 이후 `apply_lora_adapters(model, args)`를 실행해 지정된 레이어에 LoRA 래퍼 삽입.
   - 베이스라인 체크포인트를 로드한 뒤, 기존 weight는 `requires_grad=False`로 설정.

3. **Optimizer 구성**
   - `setup_optimizer_lora(args, model)`를 별도로 정의해 LoRA 파라미터만 포함하는 `AdamW` 인스턴스를 생성.
   - 학습률과 스케줄은 LoRA 전용 값을 사용(`args.lora_lr`, `args.lora_warmup`).

4. **학습 루프**
   - `train_step_lora` 함수로 분리하여 forward/backward 시 LoRA 파라미터만 업데이트.
   - 체크포인트 저장 시 LoRA state dict만 디스크에 저장(`models_lora/` 디렉터리).

5. **추론/배포**
   - `infer.py`에 `--use-lora` 옵션 추가 → 추론 시 LoRA weight를 로드해 합산하거나, merge한 가중치를 생성.

이 방식으로 베이스라인 학습 스크립트를 유지하면서 LoRA 전용 경량화 실험을 명확히 분리할 수 있다.

### LoRA 실험 절차 (베이스라인 대비 비교 포함)

1. **준비**
   - 저장소 최신 동기화(`git pull`).
   - 베이스라인 학습 완료 여부 확인(`experiment/baseline/models/checkpoint_*.tar`). 없으면 `examples/train.yaml`로 학습.
   - Python 환경 `seg_eend` 활성화 및 데이터 준비.

2. **LoRA 코드 통합**
   - `common_utils/lora.py` 작성 → `LoRALinear` 구현.
   - `backend/models.py`에 `apply_lora_adapters(model, args)` 추가.
   - `backend/updater.py`에서 LoRA 파라미터 전용 optimizer 그룹 구성.
   - 체크포인트 저장/로드 경로 확장(LoRA state 포함).
   - CLI/YAML 인자(`use_lora`, `lora_rank`, `lora_alpha`, `lora_dropout`, `lora_lr`) 추가.

3. **LoRA 전용 학습 스크립트**
   - `eend/train_lora.py` 생성: `train.py` 기반으로 `--use_lora` 플래그를 기본 활성화.
   - 베이스라인 checkpoint 로드 후 LoRA 파라미터만 `requires_grad=True`로 설정.
   - LoRA 전용 optimizer(`setup_optimizer_lora`)와 학습 루프(`train_step_lora`) 적용.

4. **하이퍼파라미터 YAML**
   - `examples/train_lora.yaml` 작성: 베이스라인과 동일한 설정에 LoRA 옵션 추가.
   - 출력 디렉터리 `experiment/lora/` 등으로 분리.

5. **LoRA 학습 실행**
   ```bash
   torchrun --nproc_per_node=1 --max_restarts=0 --standalone --tee 3 \
     eend/train_lora.py -c examples/train_lora.yaml --noam-k 1 \
     --train-batchsize 48 --accum-steps 1 --prefetch-factor 1 \
     --multiprocessing-context fork
   ```
   - Loss, DER, GPU utilization 모니터링.
   - 학습 완료 후 `experiment/lora/models/`에 checkpoint 저장 확인.

6. **파라미터 수/모델 크기 비교**
   - 베이스라인: `backend/models.get_model`로 로드해 총 파라미터 수 계산, checkpoint 파일 크기 확인.
   - LoRA: 동일 방식 + LoRA trainable params 출력. YAML/스크립트에 `print_trainable_params_lora()` 유틸 추가.

7. **성능 평가**
   - 베이스라인 추론: `python eend/infer.py -c examples/infer.yaml`.
   - LoRA 추론: `python eend/infer.py -c examples/infer_lora.yaml --use-lora`.
   - DER/기타 지표를 표로 정리.

8. **문서화 및 커밋**
   - `docs/lightweight_results.md` 작성: 조건, 파라미터 수, 모델 크기, DER 비교, GPU 사용률 등 기록.
   - 변경 파일(`common_utils/lora.py`, `backend/models.py`, `train_lora.py`, YAML 등) 커밋/푸시.

이 절차를 따르면 베이스라인 대비 LoRA 경량화 모델의 파라미터 수, 모델 크기, 성능을 체계적으로 비교할 수 있다.

## 3. 양자화 (Quantization)
- **대상**: Inference throughput 개선. Linear, Conv1d (있다면), FFN에 적용.
- **Post-Training Quantization (PTQ)**
  1. `backend/models.py`에 양자화 준비 함수 `def to_dynamic_int8(model, modules=('Linear',))` 추가 → `torch.quantization.quantize_dynamic` 호출.
  2. `infer.py`에서 체크포인트 로드 후 `to_dynamic_int8` 적용하도록 옵션 추가 (`--quantize int8`).
  3. 검증용 캘리브레이션 스크립트 작성(소량의 dev 데이터를 통과시켜 정확도 확인).
- **Quantization-Aware Training (QAT)**
  1. `backend/models.py`에 `prepare_for_qat` 구현: `torch.quantization.prepare_qat(model, qconfig_dict)` 호출.
  2. `train.py`에서 QAT 플래그가 켜지면 학습 시작 전 호출, 학습 종료 후 `torch.quantization.convert`로 변환하여 저장.
  3. YAML에 `use_qat`, `qat_backend`, `qat_start_epoch` 등을 추가.
- **유의 사항**: CUDA STFT/로그멜 경로는 float32 기준이므로, 양자화된 모델과 호환되는지 테스트 필요. PTQ/QAT 결과 DER 비교 필수.

## 종합 일정
1. 코드 베이스 분기 생성: `feature_stage=cuda` 기반 유지.
2. LoRA 적용 → 테스트 → 문서화.
3. 프루닝 적용 → 후속 fine-tuning.
4. PTQ/QAT 적용 및 성능 측정.

본 계획은 모두 코드 수정이 필수이며, YAML 수정만으로는 구현되지 않는다.
