# DeepVoice Detection

[English README](README.md)

DACON 딥보이스 탐지 대회를 위한 오프라인 추론 파이프라인입니다. 기존 노트북은
베이스라인 참고용으로 유지하며, 제출 실행 파일은 `script.py`입니다.

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
│   └── build_submissions.py      # fusion별 DACON zip 생성
├── model/                        # 로컬 모델 파일, Git 제외
└── data/                         # 대회 입력 데이터, Git 제외
```

사용 가능한 fusion 방식은 `baseline`, `component_max`, `soft_or`입니다.
