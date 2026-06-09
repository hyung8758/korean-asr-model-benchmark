## Install

이 문서는 이 저장소를 clone한 사용자가 benchmark 실행 환경과 모델/엔진을 준비하는 방법을 정리한다.

### Python Environment

Python 3.10 이상 환경을 권장한다.

```bash
pip install -r requirements.txt
```

현재 `requirements.txt`는 CUDA 12.8용 PyTorch wheel을 기준으로 한다.

```text
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
```

다른 CUDA 환경을 사용하면 먼저 PyTorch와 torchaudio 버전을 환경에 맞게 바꾼다.

### OpenAI Whisper

설치:

```bash
pip install openai-whisper
```

모델은 실행 시 자동으로 다운로드된다. config에서는 `small`, `medium`, `large-v3` 같은 model id를 사용한다.

실행:

```bash
python scripts/run_openai_whisper_baseline.py
```

### faster-whisper

설치:

```bash
pip install faster-whisper
```

모델은 실행 시 자동으로 다운로드된다. config에서는 `small`, `medium`, `large-v3` 같은 model id를 사용한다.

실행:

```bash
python scripts/run_faster_whisper.py
```

### whisper.cpp

whisper.cpp는 git submodule로 포함되어 있다. 저장소를 clone한 뒤 submodule을 초기화한다.

```bash
git submodule update --init --recursive
```

이 benchmark의 whisper.cpp server runner는 backend inference time을 `timings.inference_sec` 필드에서 읽는다. 이 필드는 upstream whisper.cpp server 기본 응답에는 없으므로 build 전에 timing patch를 적용한다.

```bash
git -C third_party/whisper.cpp apply ../../patches/whisper_cpp_server_timings.patch
```

이 patch는 자동으로 적용되지 않는다. `git submodule update --init --recursive`는 upstream whisper.cpp를 가져오는 명령이고, patch 적용은 위 명령으로 별도 실행해야 한다. submodule을 업데이트하거나 초기화한 뒤에는 patch를 다시 적용한다.

CUDA build:

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -DGGML_CUDA=ON
cmake --build third_party/whisper.cpp/build -j
```

CPU only build:

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -DGGML_CUDA=OFF
cmake --build third_party/whisper.cpp/build -j
```

모델은 `third_party/whisper.cpp/models/` 아래에 다운로드한다.

```bash
cd third_party/whisper.cpp

./models/download-ggml-model.sh tiny-q5_1
./models/download-ggml-model.sh base-q5_1
./models/download-ggml-model.sh small-q5_1
./models/download-ggml-model.sh medium-q5_0
./models/download-ggml-model.sh large-v3
./models/download-ggml-model.sh large-v3-q5_0
./models/download-ggml-model.sh large-v3-turbo
./models/download-ggml-model.sh large-v3-turbo-q5_0

cd ../..
```

기본 config는 아래 경로를 기준으로 한다.

```text
third_party/whisper.cpp/build/bin/whisper-server
third_party/whisper.cpp/models/ggml-*.bin
```

실행:

```bash
python scripts/run_whisper_cpp_server.py
```

CPU config:

```bash
python scripts/run_whisper_cpp_server.py \
  --config configs/whisper_cpp_server_cpu_experiments.json
```

### Data Preparation

이 repository는 원본 corpus와 정제된 benchmark wav를 포함하지 않는다. 사용자는 corpus를 직접 준비한 뒤 `data/` 아래에 배치한다.

```text
data/
  Zeroth Korean/
  Pansori-TEDxKR/
  ASR-KCSC Korean Conversational Speech Corpus/
```

디렉토리 이름은 정확히 일치하지 않아도 된다. prepare script는 lower-case 기준으로 다음 키워드를 이용해 corpus를 자동 추정한다.

```text
zeroth, zeroth_korean
pansori, tedx, tedxkr
kcsc, conversational, asr-kcsc
```

benchmark manifest 생성:

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
    <utt_id>.wav
  manifest.jsonl
  summary.json
  dropped_samples.jsonl
```

`benchmark_data/`는 `.gitignore`에 포함되어 있으며 GitHub에 업로드하지 않는다.

### Hugging Face ASR Models

대상 모델:

```text
Qwen/Qwen3-ASR-1.7B
ghost613/whisper-large-v3-turbo-korean
nyrahealth/CrisperWhisper
```

설치:

```bash
pip install transformers accelerate qwen-asr
```

모델은 실행 시 Hugging Face cache에 자동으로 다운로드된다. cache 위치를 고정하고 싶으면:

```bash
export HF_HOME=/path/to/hf_cache
export HF_HUB_CACHE=/path/to/hf_cache/hub
```

실행:

```bash
python scripts/run_hf_asr_models.py
```

기본 설정은 `configs/hf_asr_experiments.json`을 사용한다.
