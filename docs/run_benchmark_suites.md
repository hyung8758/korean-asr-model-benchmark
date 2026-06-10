## 실험 묶음 실행

실험 묶음 설정은 여러 runner를 한 번에 실행하기 위한 파일이다.

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
