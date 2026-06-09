## Korean Whisper STT Benchmark

한국어 utterance benchmark용 manifest를 만들고 Whisper 계열 엔진을 같은 포맷으로 비교한다.

### 설치

```bash
pip install -r requirements.txt
```

### 데이터 준비

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

### 엔진 실행

OpenAI Whisper:

```bash
python scripts/run_openai_whisper_baseline.py
```

faster-whisper:

```bash
python scripts/run_faster_whisper.py
```

whisper.cpp server CUDA:

```bash
cd third_party/whisper.cpp
cmake --build build -j
cd ../..
python scripts/run_whisper_cpp_server.py
```

whisper.cpp server CPU:

```bash
python scripts/run_whisper_cpp_server.py \
  --config configs/whisper_cpp_server_cpu_experiments.json
```

CPU 기본값은 server worker 1개, worker당 thread 8개다. worker 수는 `--jobs` 또는 `launcher.num_jobs`, worker당 thread 수는 `--threads` 또는 `decode_defaults.threads`로 조정한다. CPU 부하를 피하려면 대략 `jobs * threads`가 사용 가능한 물리/논리 코어 수를 크게 넘지 않게 둔다.

```bash
python scripts/run_whisper_cpp_server.py \
  --config configs/whisper_cpp_server_cpu_experiments.json \
  --jobs 2 \
  --threads 8
```

작게 테스트:

```bash
python scripts/run_whisper_cpp_server.py \
  --experiment small_q5_1_beam1_server \
  --limit 10 \
  --result_root results_dev
```

### 결과 구조

```text
results/<engine>/<model>/<experiment>/
  predictions.jsonl
  errors.jsonl
  metrics.json
  run_config.json
```

`predictions.jsonl`에는 sample별 reference, prediction, segment timestamp, decode_time, RTF가 저장된다. `metrics.json`에는 전체/dataset/bucket별 CER, WER, RTF가 저장된다. CER/WER는 퍼센트 스케일이다.

HF 강화 모델:

```bash
python scripts/run_hf_asr_models.py
```

기본 설정은 `configs/hf_asr_experiments.json`이며 CrisperWhisper, ghost613 한국어 Whisper turbo, Qwen3-ASR-1.7B를 실행한다. `benchmark_data/manifest.jsonl`을 입력으로 사용하고 같은 결과 포맷으로 저장한다.

### whisper.cpp server timing

server runner는 HTTP 왕복 시간이 아니라 `whisper-server` 내부 `timings.inference_sec`로 `decode_time`과 RTF를 계산한다. 그래서 `third_party/whisper.cpp/examples/server/server.cpp` timing patch를 반영해 rebuild해야 한다. HTTP 왕복 시간은 prediction row의 `request_time`에 별도 저장된다.
