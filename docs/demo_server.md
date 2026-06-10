## 데모 서버

브라우저에서 마이크 녹음 또는 음성 파일 업로드를 통해 여러 STT 엔진의 결과와 처리 시간을 한 화면에서 비교하는 데모 앱이다.

### 구성

```text
demo/
  backend/
    app/
  frontend/
    src/
```

backend는 FastAPI, frontend는 Vite/React로 구성되어 있다.

### 화면 동작

- 왼쪽 모델 row를 클릭하면 해당 엔진이 선택/해제된다.
- 파일 드랍존에 음성을 올리고 `인식`을 누르면 선택된 엔진으로 offline 인식을 수행한다.
- 마이크 버튼을 누르면 녹음이 시작되고, 다시 누르면 중지된다.
- `Offline / Streaming` 스위치는 같은 위치를 눌러 모드를 전환한다.
- 녹음 중에는 모드 전환과 파일 업로드 인식이 잠긴다.
- 새 녹음이나 새 파일 인식을 시작하면 이전 요청은 frontend에서 즉시 취소되고, 늦게 도착한 이전 결과는 화면에 반영하지 않는다.
- 이미 backend 디코더에 들어간 작업은 모델 라이브러리 특성상 강제 중단하지 않고 끝까지 실행될 수 있다. 대신 같은 엔진의 디코딩은 backend에서 직렬화해 모델 객체에 요청이 겹치지 않게 한다.
- 각 row에는 인식 결과, audio duration, decode time, total time, RTF가 표시된다.

### 통합 실행

처음 한 번은 데모 실행 의존성을 설치한다.

```bash
conda activate korean-asr-benchmark
conda install -c conda-forge nodejs -y
pip install -r demo/backend/requirements.txt
cd demo/frontend
npm install
cd ../..
```

기본 실행은 console 모드다. backend와 frontend 로그가 현재 터미널에 같이 출력되고, `Ctrl-C`로 종료한다.

```bash
scripts/run_demo.sh
```

백그라운드 실행:

```bash
scripts/run_demo.sh start
```

종료:

```bash
scripts/run_demo.sh stop
```

재시작:

```bash
scripts/run_demo.sh restart
```

상태 확인:

```bash
scripts/run_demo.sh status
```

기본 backend는 `gunicorn + uvicorn worker`로 실행한다. WebSocket을 포함한 ASGI 앱을 안정적으로 띄우기 위해 이 방식을 사용한다.

환경변수:

```text
DEMO_CONDA_ENV=korean-asr-benchmark
DEMO_BACKEND_HOST=0.0.0.0
DEMO_BACKEND_PORT=16000
DEMO_FRONTEND_HOST=0.0.0.0
DEMO_FRONTEND_PORT=16010
DEMO_GUNICORN_WORKERS=1
DEMO_RUNTIME_DIR=demo/.runtime
DEMO_LOG_DIR=demo/.runtime/logs
DEMO_SAVE_DIR=demo/.runtime/saved_audio
DEMO_START_WHISPER_CPP=1
DEMO_WHISPER_CPP_PORT=8100
DEMO_WHISPER_CPP_DEVICE_INDEX=2
DEMO_WHISPER_CPP_FLASH_ATTN=0
DEMO_WHISPER_CPP_MODEL=large-v3-q5_0
DEMO_WHISPER_CPP_MODEL_PATH=third_party/whisper.cpp/models/ggml-large-v3-q5_0.bin
DEMO_STREAM_PARTIAL_MIN_SECONDS=1.0
DEMO_STREAM_PARTIAL_INTERVAL_SECONDS=1.0
DEMO_STREAM_PARTIAL_WINDOW_SECONDS=20.0
```

backend가 시작되면 Python 기반 모델은 백그라운드에서 미리 로딩된다. 화면에는 `모델 로딩 중`, `준비 완료`, `인식 중` 상태가 표시된다. `DEMO_GUNICORN_WORKERS`를 늘리면 worker마다 모델을 따로 로딩할 수 있으므로, 데모 비교 용도에서는 기본값 `1`을 유지한다.

offline 모드에서 여러 엔진을 선택하면 같은 음성을 선택된 엔진들에 병렬 요청한다. 각 모델을 서로 다른 GPU에 올려두면 한 번의 녹음 또는 업로드로 여러 엔진의 처리 시간을 빠르게 비교할 수 있다.

streaming 모드는 실제 streaming decoder가 아니라 pseudo-streaming이다. frontend가 기본 1000ms 단위로 wav chunk를 보내고, backend가 일정 시간마다 최근 window를 기존 offline decoder에 넣어 partial 결과를 만든다. stop 이후에는 전체 녹음본을 다시 인식해 final 결과를 반환한다.

기본값은 첫 partial을 빠르게 보기 위한 설정이다.

```text
DEMO_STREAM_PARTIAL_MIN_SECONDS=1.0
DEMO_STREAM_PARTIAL_INTERVAL_SECONDS=1.0
DEMO_STREAM_PARTIAL_WINDOW_SECONDS=20.0
```

값을 바꾸려면 backend를 다시 시작해야 한다.

```bash
DEMO_STREAM_PARTIAL_MIN_SECONDS=0.5 \
DEMO_STREAM_PARTIAL_INTERVAL_SECONDS=1.0 \
bash scripts/run_demo.sh restart
```

chunk 전송 단위는 frontend의 `DEFAULT_OPTIONS.chunkMs`에서 조정한다. 기본값은 `1000`이다.

기본 GPU 배치는 아래와 같다. 필요하면 환경변수로 바꾼다.

```text
DEMO_OPENAI_WHISPER_DEVICE=cuda:0
DEMO_FASTER_WHISPER_DEVICE=cuda:1
DEMO_WHISPER_CPP_DEVICE_LABEL=cuda:2/server
DEMO_QWEN_DEVICE=cuda:3
DEMO_CRISPERWHISPER_DEVICE=cuda:4
DEMO_GHOST613_DEVICE=cuda:5
```

### 로그와 저장 음성

디코딩 요청에 사용된 음성은 기본적으로 아래 위치에 저장된다.

```text
demo/.runtime/saved_audio/
```

디코딩 로그는 jsonl 형식으로 저장된다.

```text
demo/.runtime/logs/decoding_events.jsonl
demo/.runtime/logs/model_events.jsonl
```

`decoding_events.jsonl`에는 요청 ID, 엔진, 모델, 저장 음성 경로, audio duration, model load time, decode time, total time, RTF, 처리량, segment 개수 등이 남는다. `model_events.jsonl`에는 lazy-load가 발생한 모델 로딩 시간이 기록된다.

`demo/.runtime/` 아래 로그와 저장 음성은 로컬 실행 산출물이며 GitHub에 포함하지 않는다.

### Backend 실행

개별 실행이 필요할 때만 아래 명령을 사용한다.

```bash
conda activate korean-asr-benchmark
pip install -r demo/backend/requirements.txt
gunicorn app.main:app \
  --chdir demo/backend \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:16000 \
  --timeout 0
```

모델은 처음 요청될 때 lazy-load된다. 따라서 첫 디코딩은 모델 로딩 시간 때문에 느릴 수 있고, 이후 같은 모델 요청은 메모리에 올라간 모델을 재사용한다.

`scripts/run_demo.sh`는 기본적으로 whisper.cpp server도 함께 띄운다. 기본 URL은 아래와 같다.

```text
http://127.0.0.1:8100/inference
```

whisper.cpp를 직접 관리하고 싶으면 자동 실행을 끄고, 사용자가 별도 터미널에서 `whisper-server`를 상주시킨다.

```bash
export DEMO_START_WHISPER_CPP=0
```

이미 다른 포트에 whisper.cpp server를 띄웠다면 URL을 바꾼다.

```bash
export DEMO_WHISPER_CPP_SERVER_URL=http://127.0.0.1:8100/inference
```

Qwen3-ASR과 일부 Hugging Face Whisper 파생 모델은 timestamp 설정이 모델별로 다르다. 데모에서는 안정적인 텍스트 디코딩을 우선해서 Qwen3-ASR과 ghost613 모델의 timestamp 요청을 기본으로 끈다. CrisperWhisper처럼 timestamp 설정이 정상 제공되는 모델은 segment 정보를 표시한다.

### Frontend 실행

```bash
cd demo/frontend
npm install
npm run dev
```

backend 주소를 바꾸려면:

```bash
VITE_API_BASE=http://127.0.0.1:16000 npm run dev
```

기본값은 현재 브라우저가 접속한 hostname의 16000번 포트를 backend로 사용한다. 예를 들어 `http://server-ip:16010`으로 접속하면 API는 `http://server-ip:16000`으로 요청된다.

### 지원 기능

- OpenAI Whisper, faster-whisper, whisper.cpp server, Qwen3-ASR, Hugging Face Transformers 모델 선택
- 마이크 녹음 후 offline 디코딩
- 음성 파일 업로드 후 offline 디코딩
- 1000ms 기본 chunk 단위 pseudo-streaming
- streaming 중 새 녹음 시작 시 이전 결과 flush
- 파일 드래그 앤 드롭 업로드
- 디코딩 시간, 전체 처리 시간, RTF, segment timestamp 표시

### 환경변수

```text
DEMO_OPENAI_WHISPER_MODEL
DEMO_OPENAI_WHISPER_DEVICE
DEMO_OPENAI_WHISPER_PRECISION
DEMO_FASTER_WHISPER_MODEL
DEMO_FASTER_WHISPER_DEVICE
DEMO_FASTER_WHISPER_COMPUTE_TYPE
DEMO_WHISPER_CPP_MODEL
DEMO_WHISPER_CPP_SERVER_URL
DEMO_WHISPER_CPP_DEVICE_LABEL
DEMO_QWEN_MODEL
DEMO_QWEN_DEVICE
DEMO_QWEN_PRECISION
DEMO_CRISPERWHISPER_MODEL
DEMO_CRISPERWHISPER_DEVICE
DEMO_GHOST613_MODEL
DEMO_GHOST613_DEVICE
DEMO_HUGGINGFACE_DEVICE
DEMO_HUGGINGFACE_PRECISION
```
