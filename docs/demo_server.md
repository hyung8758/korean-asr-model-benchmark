## 데모 서버

브라우저에서 여러 STT 엔진의 인식 결과와 처리 시간을 비교하는 데모 앱이다. 각 엔진은 독립 worker process에서 실행된다.

### 설치

데모 실행 전에 [install.md](install.md)의 conda 환경, submodule 초기화, patch 적용, backend/frontend 의존성 설치를 완료한다.

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

### 설정

설정은 `demo/config.yaml`에서 수정하고 재시작한다. 아래 값들은 예시이며, 실제 포트와 GPU 번호는 실행 환경에 맞게 조정한다.

외부 Chrome에서 마이크를 쓰려면 HTTPS/WSS가 필요하다. 외부에는 frontend 포트만 열고, backend와 whisper.cpp 포트는 내부에서만 사용한다.

```yaml
server:
  backend_host: 127.0.0.1
  backend_port: 16000
  frontend_host: 0.0.0.0
  frontend_port: 16010
  ssl:
    enabled: 1
    auto_generate: 1
```

접속 예시:

```text
https://서버_IP_또는_도메인:<frontend_port>
```

업로드 크기, 음성 길이, 동시 실행 수는 `server.security`에서 조정한다. 사용할 GPU는 `resources.gpu_indices`에서 정한다. GPU 개수를 초과한 모델은 비활성화 상태로 시작한다.

```yaml
resources:
  gpu_indices: [0, 1]
```

위 설정이면 첫 번째와 두 번째 모델만 로딩되고, 나머지는 비활성화된다. 비활성 모델을 켜려면 먼저 다른 모델을 내려 GPU를 비워야 한다.

`theme: streaming`으로 지정된 엔진은 streaming 전용 모델/도구로 표시되며, 화면에서 초록색 row로 구분된다.

worker 내부 포트는 `workers.base_port`부터 엔진 순서대로 사용한다. 시작 시 동시 로딩 수는 `workers.parallel_preload`로 정한다.

```yaml
workers:
  base_port: 17000
  parallel_preload: 3
```

`server.gunicorn_workers`는 `1`로 둔다. 모델별 독립 process는 내부 worker가 담당한다.

### 주요 기능

- OpenAI Whisper, faster-whisper, whisper.cpp, Qwen3-ASR, Hugging Face 모델, Whisper-Streaming, SimulStreaming 비교
- Silero VAD 기반 발화 단위 인식
- offline / pseudo-streaming 모드
- Whisper-Streaming / SimulStreaming native streaming 세션
- 모델별 결과, audio duration, decode time, total time, RTF 표시
- 모델 활성화/비활성화 및 row 순서 변경
- 변경한 row 순서는 브라우저 localStorage에 저장
- 오류와 제한 초과 메시지 팝업

### 로그

실행 로그와 저장 음성은 `logs/YYYYMMDD_HHMMSS_log/`에 생성된다. `logs/current_log/`는 최신 로그 디렉토리를 가리킨다.

```text
logs/current_log/
```

pid 파일은 `logs/.pid/`에 저장된다.
