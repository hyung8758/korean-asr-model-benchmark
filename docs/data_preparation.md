## 데이터 준비

원본 corpus, split cache, benchmark/fine-tuning 데이터는 GitHub에 포함하지 않는다. 실데이터는 모두 `data/` 아래에 둔다.

```text
data/
  download/
    zeroth_korean.tar.gz
    pansori-tedxkr-corpus-1.0.tar.gz
    Korean_Conversational_Speech_Corpus.zip
    extracted/
      zeroth_korean/
      pansori-tedxkr-corpus-1.0/
      Korean_Conversational_Speech_Corpus/
    splits/
      asr_kcsc/
  benchmark/
    manifest.jsonl
    summary.json
    dropped_samples.jsonl
```

### Benchmark 데이터

Benchmark 데이터는 raw corpus에서 직접 만든다.

```bash
python scripts/data/prepare_whisper_benchmark_data.py \
  --data_root ./data/download \
  --output_root ./data/benchmark \
  --max_hours_per_corpus 10 \
  --seed 42
```

`data/download` 아래 archive가 있으면 `data/download/extracted`로 자동 압축 해제한 뒤 corpus를 탐색한다. 긴 원본 wav에서 일부 구간만 쓰는 corpus는 `data/download/splits/<dataset>/` 아래에 16k mono wav를 한 번 생성하고 이후 재사용한다.

### Fine-Tuning 데이터

학습용 데이터도 benchmark manifest를 재사용하지 않고 raw corpus에서 직접 만든다.

```bash
python scripts/data/prepare_whisper_finetuning_data.py \
  --data_root ./data/download \
  --output_root ./domain_finetuning/data/whisper_small_lora \
  --seed 42
```

생성이 끝나면 `train/dev/eval` 파일 구조와 오디오 경로를 자동으로 검증한다.

출력:

```text
domain_finetuning/data/whisper_small_lora/
  train.jsonl
  dev.jsonl
  eval.jsonl
  metadata/
    summary.json
    dropped_samples.jsonl
```

학습용 jsonl은 STT 학습에 필요한 최소 필드만 저장한다.

```json
{"id": "sample_id", "audio": "path/to/audio.wav", "text": "정규화된 정답 문장"}
```

기본값은 1초 이상 30초 이하 utterance만 사용하고, dataset/bucket 기준으로 train/dev/eval을 8:1:1 비율로 deterministic split한다. `dev.jsonl`은 학습 중 validation에, `eval.jsonl`은 최종 성능 평가에 사용한다.

`metadata/summary.json`은 생성된 split별 샘플 수, 시간, dataset/bucket 통계, 제거 사유별 개수를 담는다. `metadata/dropped_samples.jsonl`은 너무 짧은 음성, 빈 transcript, 로딩 실패처럼 사용하지 않은 샘플을 reason과 함께 기록한다.

### 설계 원칙

- benchmark 데이터와 fine-tuning 데이터는 서로의 manifest를 입력으로 사용하지 않는다.
- 둘 다 raw/extracted corpus를 직접 읽어서 생성한다.
- 학습용 jsonl의 `audio`는 실제 사용할 오디오를 가리킨다.
- 긴 원본에서 잘라낸 wav는 `data/download/splits/`에 한 번만 저장하고 benchmark/fine-tuning에서 같이 참조한다.
- 원본 추적용 상세 정보는 학습 jsonl에 넣지 않고 `metadata/`, prepare log에서 확인한다.
- 나중에 도메인 corpus가 추가되면 raw parser 또는 별도 data preparation을 추가해서 같은 manifest 형식으로 맞춘다.
