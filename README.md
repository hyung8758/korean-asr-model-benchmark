## 한국어 STT 모델 벤치마크

공개 STT 모델과 Whisper 계열 엔진을 같은 한국어 벤치마크 manifest로 실행하고, CER/WER, RTF, 디코딩 오류 패턴을 비교하기 위한 저장소다.

### 비교 대상

- OpenAI Whisper
- faster-whisper
- whisper.cpp server
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

```bash
python scripts/prepare_whisper_benchmark_data.py \
  --data_root ./data \
  --output_root ./benchmark_data \
  --sample_rate 16000 \
  --max_hours_per_corpus 10 \
  --seed 42

python scripts/run_benchmark_suite.py \
  --suite configs/suites/whisper_engine_comparison.json
```

### 데모 실행

브라우저에서 마이크 녹음 또는 음성 파일 업로드로 여러 STT 엔진의 결과와 처리 시간을 한 화면에서 비교할 수 있다.

```bash
bash scripts/run_demo.sh
```

기본 포트:

```text
Backend:  http://127.0.0.1:16000
Frontend: http://127.0.0.1:16010
```

offline 모드는 녹음/업로드 후 한 번에 인식하고, streaming 모드는 1초 chunk를 이용한 pseudo-streaming으로 중간 결과를 갱신한다. 자세한 설정은 [docs/demo_server.md](docs/demo_server.md)를 참고한다.

원본 데이터, 정제된 벤치마크 데이터, 결과물은 GitHub에 포함하지 않는다.
