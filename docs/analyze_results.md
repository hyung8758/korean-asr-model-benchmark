## 결과 분석

기본 성능 지표 생성:

```bash
python scripts/evaluate_predictions.py \
  --manifest_path benchmark_data/manifest.jsonl \
  --result_dir results/<engine>/<model>/<experiment>
```

예측 결과 형식 검증:

```bash
python scripts/validate_predictions.py \
  --manifest_path benchmark_data/manifest.jsonl \
  --predictions results/<engine>/<model>/<experiment>/predictions.jsonl
```

품질 이슈 분석:

```bash
python scripts/analyze_predictions.py \
  --predictions results/<engine>/<model>/<experiment>/predictions.jsonl
```

생성 파일:

```text
validation.json
quality_analysis.json
quality_examples.json
```

`quality_analysis.json`은 빈 출력, 비정상 길이, 반복 패턴, 자막 문구, 외국어 문자를 count/percent로 정리한다.
