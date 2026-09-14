# DeepVoice Detection

DACON 딥보이스 탐지 대회를 위한 오프라인 추론 파이프라인입니다. 기존 노트북은
베이스라인 참고용으로 유지하며, 제출 실행 파일은 `script.py`입니다.

기준 구현은 `open/baseline_submit.zip`에 포함된 공식 `script.py`입니다. 현재
`script.py`와 `deepvoice/`는 그 동작을 모듈화하고 세 가지 fusion 실험을 추가한
Git 추적 대상 코드입니다. 노트북은 설명·비교용 원본으로만 유지합니다.

## 빠른 시작

로컬 모델과 대회 입력 데이터를 아래 구조에 둡니다.

```text
model/
  df_arena_1b/
  htdemucs/
  panns/
data/
  test/
  sample_submission.csv
```

이 저장소는 모델 가중치나 공식 baseline zip을 포함하지 않습니다. Windows/Linux
학습 환경에서 공식 baseline 패키지의 `model/`을 프로젝트 루트에 배치하고 무결성을
확인합니다.

```bash
python scripts/prepare_local_assets.py --prepare-data-dirs --check-model-hashes
```

공식 모델의 출처·고정 revision·SHA-256은
[`configs/baseline_model_manifest.json`](configs/baseline_model_manifest.json)에
기록되어 있습니다.

`model/`, `data/`, `open/`, 결과물은 모두 Git에서 제외됩니다. Tailscale로 받은
학습 데이터는 우선 `data/incoming/`에 두고, 학습용 데이터를 `data/train/`, 검증용
데이터를 `data/validation/`에 배치합니다. 대회 추론 입력은 `data/test/` 및
`data/sample_submission.csv`를 사용합니다.

무거운 모델 추론을 한 번만 수행하면서 세 fusion 결과를 모두 만들려면 다음을 실행합니다.

```bash
python script.py --all-fusions
```

생성 파일은 다음과 같습니다.

```text
output/submission_baseline.csv
output/submission_component_max.csv
output/submission_soft_or.csv
```

하나의 방식으로 대회 형식의 `output/submission.csv`를 만들려면 다음처럼 실행합니다.

```bash
python script.py --fusion baseline
python script.py --fusion component_max
python script.py --fusion soft_or
```

`model/`을 채운 후 세 가지 업로드용 압축 파일을 만들 수 있습니다.

```bash
python scripts/build_submissions.py
```

## 구조

```text
.
├── [Baseline_Inference]_*.ipynb  # 원본 베이스라인 노트북
├── script.py                     # DACON 제출 실행 진입점
├── deepvoice/                    # 재사용 가능한 추론 모듈
│   ├── audio.py                  # 오디오 로딩·세그먼트 분할
│   ├── presence.py               # PANNs 존재 확률 추론
│   ├── separation.py             # HTDemucs 음원 분리
│   ├── detector.py               # DF-Arena fake 확률 추론
│   ├── fusion.py                 # 파일 단위 fusion 세 방식
│   ├── pipeline.py               # 전체 추론 흐름
│   └── metrics.py                # 로컬 ADS/CPS 지표 계산
├── scripts/
│   ├── build_submissions.py      # fusion별 DACON zip 생성
│   └── prepare_local_assets.py   # 로컬 모델·데이터 레이아웃 준비와 검증
├── configs/
│   └── baseline_model_manifest.json  # 공식 모델 메타데이터·무결성 해시
├── model/                        # 로컬 모델 파일, Git 제외
├── data/                         # Tailscale 학습 데이터·대회 입력, Git 제외
│   ├── incoming/                 # 수신 직후 원본 데이터
│   ├── train/                    # 학습 데이터
│   ├── validation/               # 검증 데이터
│   └── test/                     # 대회 추론 오디오
└── open/                         # 공식 baseline zip 보관, Git 제외
```

사용 가능한 fusion 방식은 `baseline`, `component_max`, `soft_or`입니다.
