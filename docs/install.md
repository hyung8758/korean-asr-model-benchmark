## 설치와 모델 준비

이 문서는 이 저장소를 클론한 사용자가 벤치마크 실행 환경과 모델/엔진을 준비하는 방법을 정리한다.

### Conda 환경

기본 실행 환경은 하나의 conda 환경으로 시작한다.

```bash
conda create -n korean-asr-benchmark python=3.10 -y
conda activate korean-asr-benchmark
pip install -r requirements.txt
```

현재 `requirements.txt`는 CUDA 12.8용 PyTorch wheel을 기준으로 한다.

```text
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
```

다른 CUDA 환경을 사용하면 먼저 PyTorch와 torchaudio 버전을 환경에 맞게 바꾼다.

Whisper LoRA 도메인 파인튜닝 파이프라인은 별도 저장소 `whisper-domain-finetuning-to-whispercpp`에서 관리한다.

데모 서버까지 같은 환경에서 실행하려면 backend 의존성을 추가로 설치한다.

```bash
conda install -c conda-forge nodejs -y
pip install -r demo/backend/requirements.txt
```

`nodejs`는 frontend 실행에 필요한 `npm`을 함께 설치한다. `gunicorn`은 `demo/backend/requirements.txt`에 포함되어 있다.
데모의 마이크 녹음 구간 분리를 위해 `silero-vad`도 backend 의존성에 포함되어 있다.

여러 엔진의 dependency가 충돌하면 이후 엔진별 conda 환경으로 분리한다. 지금 기본 문서는 단일 `korean-asr-benchmark` 환경을 기준으로 한다.

### OpenAI Whisper

설치:

```bash
pip install openai-whisper
```

모델은 실행 시 자동으로 다운로드된다. 설정 파일에서는 `small`, `medium`, `large-v3` 같은 모델 id를 사용한다.

실행:

```bash
python scripts/run_openai_whisper_baseline.py
```

### faster-whisper

설치:

```bash
pip install faster-whisper
```

모델은 실행 시 자동으로 다운로드된다. 설정 파일에서는 `small`, `medium`, `large-v3` 같은 모델 id를 사용한다.

실행:

```bash
python scripts/run_faster_whisper.py
```

### Whisper-Streaming

Whisper-Streaming은 git submodule로 포함되어 있다. clone 후 submodule을 초기화하면 `third_party/whisper_streaming`에 upstream 코드가 준비된다.

```bash
git submodule update --init --recursive
pip install -r demo/backend/requirements.txt
```

데모에서는 `third_party/whisper_streaming/whisper_online.py`의 `FasterWhisperASR`와 `OnlineASRProcessor`를 사용한다.

### whisper.cpp

whisper.cpp는 git submodule로 포함되어 있다. 저장소를 clone한 뒤 submodule을 초기화한다.

```bash
git submodule update --init --recursive
```

이 벤치마크의 whisper.cpp server runner는 backend inference time을 `timings.inference_sec` 필드에서 읽는다. 이 필드는 upstream whisper.cpp server 기본 응답에는 없으므로 build 전에 timing patch를 적용한다.

```bash
git -C third_party/whisper.cpp apply ../patches/whisper_cpp_server_timings.patch
```

이 patch는 자동으로 적용되지 않는다. `git submodule update --init --recursive`는 upstream whisper.cpp를 가져오는 명령이고, patch 적용은 위 명령으로 별도 실행해야 한다. submodule을 업데이트하거나 초기화한 뒤에는 patch를 다시 적용한다.

CUDA 빌드:

```bash
cmake -S third_party/whisper.cpp -B third_party/whisper.cpp/build -DGGML_CUDA=ON
cmake --build third_party/whisper.cpp/build -j
```

CPU 전용 빌드:

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

기본 설정 파일은 아래 경로를 기준으로 한다.

```text
third_party/whisper.cpp/build/bin/whisper-server
third_party/whisper.cpp/models/ggml-*.bin
```

실행:

```bash
python scripts/run_whisper_cpp_server.py
```

CPU 설정:

```bash
python scripts/run_whisper_cpp_server.py \
  --config configs/engines/whisper_cpp_server_cpu_experiments.json
```

### 데이터 준비

자세한 데이터 준비 명령은 [data_preparation.md](data_preparation.md)에 정리되어 있다.

이 저장소는 원본 corpus와 생성된 manifest/cache를 포함하지 않는다. 사용자는 corpus를 직접 준비한 뒤 `data/download/` 아래에 둔다.

```text
data/
  download/
    zeroth_korean.tar.gz
    pansori-tedxkr-corpus-1.0.tar.gz
    Korean_Conversational_Speech_Corpus.zip
    extracted/
    splits/
```

디렉토리 이름은 정확히 일치하지 않아도 된다. prepare script는 lower-case 기준으로 다음 키워드를 이용해 corpus를 자동 추정한다.

```text
zeroth, zeroth_korean
pansori, tedx, tedxkr
kcsc, conversational, asr-kcsc
```

벤치마크 manifest 생성:

```bash
python scripts/data/prepare_whisper_benchmark_data.py \
  --data_root ./data/download \
  --output_root ./data/benchmark \
  --max_hours_per_corpus 10 \
  --seed 42
```

출력:

```text
data/benchmark/
  manifest.jsonl
  summary.json
  dropped_samples.jsonl
```

`data/benchmark/`는 `.gitignore`에 포함되어 있으며 GitHub에 업로드하지 않는다.
구간 분할이 필요한 오디오는 `data/download/splits/`에 한 번 생성한 뒤 재사용한다.

### Hugging Face Transformers 모델

대상 모델:

```text
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
python scripts/run_huggingface_transformers.py
```

기본 설정은 `configs/engines/huggingface_transformers_experiments.json`을 사용한다.

### Qwen Speech Recognition

대상 모델:

```text
Qwen/Qwen3-ASR-1.7B
```

설치:

```bash
pip install qwen-asr transformers accelerate
```

실행:

```bash
python scripts/run_qwen_speech_recognition.py
```

기본 설정은 `configs/engines/qwen_speech_recognition_experiments.json`을 사용한다.
