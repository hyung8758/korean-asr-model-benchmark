## 데이터 준비

원본 corpus는 GitHub에 포함하지 않는다. 아래처럼 `data/` 아래에 배치한다.

```text
data/
  Zeroth Korean/
  Pansori-TEDxKR/
  ASR-KCSC Korean Conversational Speech Corpus/
```

벤치마크 manifest 생성:

```bash
python scripts/prepare_whisper_benchmark_data.py \
  --data_root ./data \
  --output_root ./benchmark_data \
  --sample_rate 16000 \
  --max_hours_per_corpus 10 \
  --seed 42
```

출력:

```text
benchmark_data/
  wavs/
  manifest.jsonl
  summary.json
  dropped_samples.jsonl
```

`benchmark_data/`는 `.gitignore`에 포함되어 있다.
