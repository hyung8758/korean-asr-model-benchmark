## 데모 서버

브라우저에서 여러 STT 엔진의 인식 결과와 처리 시간을 비교하는 데모 앱이다. 마이크 녹음과 음성 파일 업로드를 지원한다.

### 설치

```bash
conda activate korean-asr-benchmark
conda install -c conda-forge nodejs -y
pip install -r demo/backend/requirements.txt

cd demo/frontend
npm install
cd ../..
```

### 실행

```bash
bash scripts/run_demo.sh
```

백그라운드 실행:

```bash
bash scripts/run_demo.sh start
bash scripts/run_demo.sh stop
bash scripts/run_demo.sh restart
```

localhost HTTP 모드:

```bash
DEMO_LOCAL_MODE=1 bash scripts/run_demo.sh
```

### 설정

기본 설정은 `demo/config.yaml`에서 수정한다. 값을 바꾼 뒤에는 데모를 재시작한다.

외부 Chrome에서 마이크를 쓰려면 HTTPS/WSS가 필요하다. 외부에는 frontend 포트만 열고, backend와 whisper.cpp server 포트는 내부에서만 사용한다.

```yaml
server:
  backend_host: 127.0.0.1
  backend_port: 16000
  frontend_host: 0.0.0.0
  frontend_port: 41301
  ssl:
    enabled: 1
    auto_generate: 1
```

접속 예시:

```text
https://서버_IP_또는_도메인:41301
```

업로드 크기, 음성 길이, 동시 실행 수는 `server.security`에서 조정한다. 기본값은 100MB, 20분, 동시 3세션이다.

### 기능

- OpenAI Whisper, faster-whisper, whisper.cpp server, Qwen3-ASR, Hugging Face 모델 비교
- Silero VAD 기반 발화 단위 인식
- offline / pseudo-streaming 모드
- 모델별 결과, audio duration, decode time, total time, RTF 표시
- 모델 활성화/비활성화 및 row 순서 변경
- 오류와 제한 초과 메시지 팝업 표시

### 로그

실행 로그와 저장 음성은 `logs/YYYYMMDD_HHMMSS_log/`에 생성된다.

```text
logs/current_log/
```

`logs/current_log/`는 최신 로그 디렉토리를 가리킨다. pid 파일은 `logs/.pid/`에 저장된다.
