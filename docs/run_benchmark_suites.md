## 실험 묶음 실행

실험 묶음 설정은 여러 runner를 한 번에 실행하기 위한 파일이다. suite는 모델/엔진별 runner를 순서대로 실행하고, 각 runner는 설정된 GPU 수만큼 manifest를 shard로 나누어 병렬 디코딩한 뒤 결과를 병합하고 평가한다.

```bash
python scripts/run_benchmark_suite.py \
  --suite configs/suites/whisper_engine_comparison.json
```

실행하지 않고 명령어만 확인:

```bash
python scripts/run_benchmark_suite.py \
  --suite configs/suites/full_benchmark.json \
  --dry_run
```

특정 엔진만 실행:

```bash
python scripts/run_benchmark_suite.py \
  --suite configs/suites/full_benchmark.json \
  --only faster_whisper_all
```

공통 override:

```bash
python scripts/run_benchmark_suite.py \
  --suite configs/suites/whisper_engine_comparison.json \
  --gpus 0,1,2,3,4,5 \
  --result_root results
```

`--gpus`는 실행 환경에 맞게 바꾼다. 예를 들어 `--gpus 0,1,2`이면 각 실험을 3개 shard로 나누어 실행한다. shard 결과는 각 result directory의 `shards/` 아래에 저장되고, 완료 후 `predictions.jsonl`, `errors.jsonl`, `metrics.json`으로 병합된다.

일부 runner는 `manifest_paths`를 읽어 임시 combined manifest를 만든 뒤 실행한다. 이 파일은 같은 입력 포맷을 유지하기 위한 실행 산출물이며, 원본 benchmark manifest는 `data/benchmark/manifest.jsonl`이다.
