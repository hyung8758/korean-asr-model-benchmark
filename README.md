## 한국어 STT 모델 벤치마크

공개 STT 모델과 Whisper 계열 엔진을 같은 한국어 벤치마크 manifest로 실행하고, CER/WER, RTF, 디코딩 오류 패턴을 비교하기 위한 저장소다.

### 비교 대상

- OpenAI Whisper
- faster-whisper
- whisper.cpp
- Whisper-Streaming
- SimulStreaming
- Hugging Face Transformers 기반 STT 모델
- Qwen Speech Recognition

### 기본 흐름

```text
manifest.jsonl -> engine decoder -> predictions.jsonl/errors.jsonl -> metrics.json
```

### 문서

- 설치와 모델 준비: [docs/install.md](docs/install.md)
- 데이터 준비: [docs/data_preparation.md](docs/data_preparation.md)
- 실험 묶음 실행: [docs/run_benchmark_suites.md](docs/run_benchmark_suites.md)
- 결과 분석: [docs/analyze_results.md](docs/analyze_results.md)
- 실험 결과 정리: [docs/experiment_results.md](docs/experiment_results.md)
- 데모 서버: [docs/demo_server.md](docs/demo_server.md)

### 빠른 실행 예시

먼저 [설치와 모델 준비](docs/install.md)를 진행한다. 이 과정에서 submodule 초기화와 필요한 patch 적용까지 함께 수행한다.

```bash
python scripts/data/prepare_whisper_benchmark_data.py \
  --data_root ./data/download \
  --output_root ./data/benchmark \
  --sample_rate 16000 \
  --max_hours_per_corpus 10 \
  --seed 42

python scripts/run_benchmark_suite.py \
  --suite configs/suites/whisper_engine_comparison.json
```

### 데모 실행

브라우저에서 마이크 녹음 또는 음성 파일 업로드로 여러 STT 엔진의 결과와 처리 시간을 한 화면에서 비교할 수 있다.
데모 실행 전에도 [docs/install.md](docs/install.md)의 conda 환경, submodule, patch, frontend 설치 단계를 먼저 완료한다.

```bash
bash scripts/run_demo.sh
```

내부/외부에서 접속할 때 frontend 포트만 연다.

```text
https://서버_IP_또는_도메인:16010
```

파일 업로드와 마이크 녹음 모두 Silero VAD로 발화 단위를 자른 뒤 offline 또는 pseudo-streaming 방식으로 결과를 갱신한다. 실행 오류와 제한 초과 메시지는 화면 팝업으로 표시된다. 자세한 설정은 [docs/demo_server.md](docs/demo_server.md)를 참고한다.

데모에서는 각 STT 엔진을 독립 worker process로 실행해 엔진 간 CUDA runtime 충돌을 줄인다.

원본 데이터, 정제된 벤치마크 데이터, 결과물은 GitHub에 포함하지 않는다.

Whisper LoRA 도메인 파인튜닝 파이프라인은 별도 저장소 `whisper-domain-finetuning-to-whispercpp`에서 관리한다.
