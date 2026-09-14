# 학습 오디오 메타데이터 양식

`clip_metadata_template.csv`를 복사해 실제 데이터셋의 단일 manifest로 사용합니다.
모든 텍스트 값은 UTF-8로 저장하고, `audio_path`는 `data/` 기준 상대 경로를 씁니다.

## 필수 라벨 규칙

| 필드 | 허용 값 | 설명 |
|---|---|---|
| `split` | `train`, `validation`, `holdout` | `group_id`가 같은 행은 하나의 split에만 둔다. |
| `origin_type` | `public_dataset`, `generated`, `transformed` | 원천의 생성·수집 형태 |
| `content_type` | `voice`, `music`, `mixed`, `neither` | 파일 구성 |
| `voice_present`, `music_present` | `0`, `1` | 성분 존재 여부 |
| `voice_label`, `music_label` | `real`, `fake`, `none` | 성분이 없으면 반드시 `none` |
| `file_label` | `real`, `fake` | 어느 성분이든 `fake`이면 `fake` |
| `license_allows_noncommercial` | `true`, `false` | 라이선스·이용조건 확인 결과 |
| `label_confidence` | `high`, `medium`, `low` | 라벨 근거의 신뢰 수준 |

## 반드시 기록할 필드

- 재현성: `sample_id`, `audio_path`, `sha256`, `parent_id`, `group_id`, `split`
- 대회 타깃: `voice_present`, `music_present`, `voice_label`, `music_label`, `file_label`
- 출처·권리: `source_dataset`, `source_item_id`, `source_url`, `source_license`, `source_access_date`
- 생성 데이터: `generator_family`, `generator_name`, `generator_version`
- 일반화 분석: `language`, `channel_domain`, `codec`, `sample_rate_hz`, `transform_chain_id`

`group_id`는 같은 화자, 같은 원곡, 같은 원본 클립, 또는 같은 생성 결과에서
파생된 변형 파일을 묶는 ID입니다. train/validation 간 누수를 막기 위해 같은
`group_id`를 서로 다른 split에 넣지 않습니다.

## 예시 행

```csv
mix_000184,train/mixed/mix_000184.flac,train,song_027,mix_source_027,ExampleSet,027,https://example.org/item/027,CC-BY-NC-4.0,true,2026-09-14,generated,mixed,1,1,fake,real,fake,tts,ExampleTTS,v2,ko,spk_042,pop,telephone,codec_opus_24k,18.42,16000,1,opus,0123...,generation_log,high,"fake voice plus real instrumental"
```

실제 데이터를 기록할 때는 예시 행을 포함하지 말고, template의 헤더만 사용합니다.
