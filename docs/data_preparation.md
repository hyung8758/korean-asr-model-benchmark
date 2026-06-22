## 데이터 준비

원본 corpus, split cache, benchmark 데이터는 GitHub에 포함하지 않는다. 실데이터는 `data/` 아래에 둔다.

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

### Benchmark 데이터 생성

```bash
python scripts/data/prepare_whisper_benchmark_data.py \
  --data_root ./data/download \
  --output_root ./data/benchmark \
  --sample_rate 16000 \
  --max_hours_per_corpus 10 \
  --seed 42
```

archive는 `data/download/extracted`로 자동 압축 해제된다. 긴 원본 wav에서 잘라낸 구간은 `data/download/splits/<dataset>/`에 16k mono wav로 저장해 재사용한다.

### 설계 원칙

- benchmark 데이터는 raw/extracted corpus에서 직접 생성한다.
- manifest의 경로 필드는 절대경로로 저장한다.
- 긴 원본에서 잘라낸 wav는 `data/download/splits/`에 한 번만 저장하고 이후 재사용한다.
- Whisper LoRA 도메인 파인튜닝 데이터 준비는 별도 저장소 `whisper-domain-finetuning-to-whispercpp`에서 관리한다.
