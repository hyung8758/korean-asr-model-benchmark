## 설치와 모델 준비

벤치마크 실행 환경과 모델/엔진 준비 방법이다.

### Conda 환경

```bash
conda create -n korean-asr-benchmark python=3.10 -y
conda activate korean-asr-benchmark
pip install -r requirements.txt
```

`requirements.txt`는 CUDA 12.8용 PyTorch wheel을 기준으로 한다.

```text
torch==2.8.0+cu128
torchaudio==2.8.0+cu128
```

다른 CUDA 환경에서는 PyTorch와 torchaudio 버전을 환경에 맞게 바꾼다.

Whisper LoRA 도메인 파인튜닝 파이프라인은 별도 저장소 `whisper-domain-finetuning-to-whispercpp`에서 관리한다.

데모 서버를 실행하려면 backend/frontend 의존성을 추가로 설치한다.

```bash
conda install -c conda-forge nodejs -y
pip install -r demo/backend/requirements.txt
cd demo/frontend
npm install
cd ../..
```

`nodejs`는 `npm`을 함께 설치한다.

### Submodule과 Patch

whisper.cpp, Whisper-Streaming, SimulStreaming은 git submodule로 포함되어 있다.

```bash
git submodule update --init --recursive
```

submodule 초기화/업데이트 뒤 patch를 적용한다.

```bash
git -C third_party/whisper.cpp apply ../patches/whisper_cpp_server_timings.patch
git -C third_party/simul_streaming apply ../patches/simul_streaming_empty_fragment_guard.patch
```

### OpenAI Whisper

```bash
pip install openai-whisper
```

모델은 실행 시 자동 다운로드된다.

```bash
python scripts/run_openai_whisper_baseline.py
```

### faster-whisper

```bash
pip install faster-whisper
```

모델은 실행 시 자동 다운로드된다.

```bash
python scripts/run_faster_whisper.py
```

### Whisper-Streaming

데모에서는 `third_party/whisper_streaming/whisper_online.py`의 `FasterWhisperASR`와 `OnlineASRProcessor`를 사용한다.

### SimulStreaming

데모에서는 Simul-Whisper AlignAtt backend를 native streaming 엔진으로 사용한다.

```text
third_party/simul_streaming/models/large-v3.pt
```

파일이 없으면 SimulStreaming 로더가 다운로드한다.

### whisper.cpp

whisper.cpp server runner는 `timings.inference_sec`를 읽는다.

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

기본 설정은 아래 경로를 기준으로 한다.

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

원본 corpus와 생성 manifest/cache는 GitHub에 포함하지 않는다.

```text
data/
  download/
    zeroth_korean.tar.gz
    pansori-tedxkr-corpus-1.0.tar.gz
    Korean_Conversational_Speech_Corpus.zip
    extracted/
    splits/
```

prepare script는 아래 키워드로 corpus를 추정한다.

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

구간 분할 wav는 `data/download/splits/`에 생성해 재사용한다.

### Hugging Face Transformers 모델

대상 모델:

```text
ghost613/whisper-large-v3-turbo-korean
nyrahealth/CrisperWhisper
```

```bash
pip install transformers accelerate qwen-asr
```

모델은 Hugging Face cache에 자동 다운로드된다. cache 고정:

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

```bash
pip install qwen-asr transformers accelerate
```

실행:

```bash
python scripts/run_qwen_speech_recognition.py
```

기본 설정은 `configs/engines/qwen_speech_recognition_experiments.json`을 사용한다.
