# 실험 결과

이 문서는 한국어 utterance 벤치마크에서 OpenAI Whisper, faster-whisper, whisper.cpp와 추가 공개 STT 모델을 비교한 주요 결과를 정리한다.

## 평가 데이터

- 낭독체: Zeroth Korean
- 강연/발표체: Pansori-TEDxKR
- 자유발화/대화체: ASR-KCSC Korean Conversational Speech Corpus

평가는 CER/WER와 decoding speed를 함께 비교했다. CER/WER는 퍼센트 스케일이며, RTF는 real-time factor이다.

## OpenAI Whisper

| experiment | CER | WER | avg RTF | decode h |
| --- | ---: | ---: | ---: | ---: |
| small_beam1_fp16 | 11.12 | 32.83 | 0.1012 | 1.204 |
| small_beam5_fp16 | 10.43 | 31.72 | 0.1459 | 1.792 |
| small_beam5_fp32 | 10.44 | 31.73 | 0.1624 | 1.901 |
| medium_beam1_fp16 | 8.71 | 27.97 | 0.1725 | 2.094 |
| medium_beam5_fp16 | 7.92 | 27.31 | 0.3038 | 3.857 |
| large-v3_beam1_fp16 | **6.85** | **24.62** | 0.2507 | 3.078 |
| large-v3_beam5_fp16 | 7.25 | 24.74 | 0.5010 | 6.461 |

### OpenAI Whisper 분석

- 모델 크기가 커질수록 전반적인 CER/WER는 개선된다.
- beam size 증가는 작은 모델에서 더 효과적이었다.
- large-v3에서는 beam5가 beam1보다 항상 좋은 것은 아니며, 전체 CER 기준으로는 beam1이 더 좋았다.
- fp16과 fp32의 정확도 차이는 거의 없었고, fp32는 속도만 느려졌다.
- 데이터별로는 Zeroth 같은 짧은 낭독체에서는 beam1이 더 안정적이고, TEDxKR/ASR-KCSC처럼 긴 문맥이나 자유발화가 있는 데이터에서는 beam5가 일부 개선을 보였다.

## faster-whisper

| experiment | CER | WER | avg RTF | decode h |
| --- | ---: | ---: | ---: | ---: |
| small_beam1_float16 | 11.13 | 32.83 | **0.0665** | 0.724 |
| small_beam5_float16 | 10.43 | 31.73 | 0.0791 | 0.863 |
| small_beam5_float32 | 10.43 | 31.72 | 0.1181 | 1.261 |
| small_beam5_int8 | 10.32 | 31.73 | 0.0823 | 0.902 |
| small_beam5_int8_float16 | 10.32 | 31.73 | 0.0823 | 0.905 |
| medium_beam1_float16 | 8.71 | 27.99 | 0.1107 | 1.172 |
| medium_beam5_float16 | 7.93 | 27.32 | 0.1281 | 1.386 |
| medium_beam5_int8_float16 | 7.97 | 27.41 | 0.1299 | 1.411 |
| large-v3_beam1_float16 | 6.86 | 24.63 | 0.1739 | 1.847 |
| large-v3_beam5_float16 | 7.01 | 24.56 | 0.1956 | 2.133 |
| large-v3_beam5_int8_float16 | 7.40 | 24.86 | 0.1933 | 2.110 |

### faster-whisper 분석

- 인식 정확도는 OpenAI Whisper와 거의 유사하다.
- 같은 모델/beam 조건에서 RTF와 decode time은 OpenAI Whisper보다 확실히 낮다.
- `int8_float16`은 CTranslate2의 혼합 연산 타입이다. 모델 weight는 int8로 양자화하고, activation/compute는 float16을 사용한다.
- 이번 실험에서는 `int8_float16`이 속도나 정확도 측면에서 뚜렷한 이점을 보이지 않았다.

## whisper.cpp

| device | experiment | CER | WER | avg RTF | decode h |
| --- | --- | ---: | ---: | ---: | ---: |
| gpu | whisper.cpp server large-v3 q5_0 beam5 | **5.76** | **24.52** | 0.2344 | 2.52 |
| gpu | whisper.cpp server large-v3 q5_0 beam1 | 5.96 | 24.61 | 0.1921 | 1.94 |
| gpu | whisper.cpp server medium q5_0 beam5 | 8.11 | 28.85 | 0.1532 | 1.69 |
| gpu | whisper.cpp server medium q5_0 beam1 | 8.84 | 29.10 | 0.1189 | 1.22 |
| gpu | whisper.cpp server small q5_1 beam5 | 10.53 | 33.27 | 0.0747 | 0.87 |
| gpu | whisper.cpp server small q5_1 beam1 | 11.34 | 33.88 | 0.0502 | 0.54 |
| cpu | tiny q5_1 beam1 CPU | 34.48 | 67.04 | 0.2385 | 2.25 |
| cpu | base q5_1 beam1 CPU | 18.12 | 43.84 | 0.4927 | 4.58 |
| cpu | small q5_1 beam1 CPU | 11.37 | 33.87 | 1.6868 | 15.40 |

### whisper.cpp 분석

- CPU 기준에서는 tiny/base의 정확도가 부족하고, small부터 비교 가능한 성능이 나온다.
- CPU small은 정확도는 어느 정도 확보되지만 RTF가 1.6868로 느리다.
- GPU server 기준에서는 large-v3 q5_0의 성능이 가장 좋았다.
- OpenAI Whisper/faster-whisper와 비교했을 때 whisper.cpp large-v3는 CER가 약 1%p 정도 낮았다.
- 특히 insertion이 크게 줄어드는 경향이 있었다. 예를 들어 faster-whisper large-v3 계열보다 insertion 비율이 낮게 나타났다.

## Large-v3 엔진 비교

| experiment | CER | WER | avg RTF |
| --- | ---: | ---: | ---: |
| whisper.cpp server large-v3 beam5 | **5.76** | **24.52** | 0.2344 |
| whisper.cpp server large-v3 beam1 | 5.96 | 24.61 | 0.1921 |
| faster large-v3 beam1 float16 | 6.86 | 24.63 | **0.1739** |
| faster large-v3 beam5 float16 | 7.01 | 24.56 | 0.1956 |
| openai large-v3 beam1 fp16 | 6.85 | 24.62 | 0.2507 |

### 종합 분석

- OpenAI Whisper와 faster-whisper의 정확도는 거의 동일하다.
- faster-whisper는 같은 정확도 수준에서 더 빠른 inference를 제공한다.
- whisper.cpp server는 large-v3 기준으로 정확도와 속도 균형이 가장 좋았다.
- whisper.cpp의 성능 우위가 quantization 때문인지 확인하기 위해 non-quantized/f16 모델도 확인했지만, faster-whisper 대비 성능 우위가 유지되었다.
- beam size는 항상 클수록 좋은 방향으로 작동하지 않았다. 특히 큰 모델에서는 beam1이 더 안정적인 경우가 있었다.

## 데이터셋별 오류 패턴

- 모델이 커질수록 substitution은 감소하는 경향이 있었다.
- insertion과 deletion은 모델 크기 증가만으로 크게 줄지는 않았다.
- ASR-KCSC에서는 deletion이 높게 나타났다. 대화체, 짧은 맞장구, 발화 겹침, filler에서 일부 음절을 빼먹는 패턴이 강했다.
- Zeroth에서는 beam 증가가 오히려 짧은 발화의 hallucination을 키우는 사례가 있었다.
- TEDxKR에서는 전체 CER는 낮지만 고유 표현이나 드문 단어에서 음운적으로 비슷한 단어로 치환되는 substitution이 보였다.
- ASR-KCSC 대화체에서는 filler, 반복, 조사, 구어체 표현을 줄이거나 문장을 정리해서 출력하는 경향이 있었다.

## Whisper 엔진 문제점

한국어를 강제했음에도 일부 엔진에서 영어를 제외한 외국어 출력이 발생했다.

| engine | total | RX | TX |
| --- | ---: | ---: | ---: |
| OpenAI Whisper | 32 / 6500, 0.49% | 25 / 3500, 0.71% | 7 / 3000, 0.23% |
| faster-whisper | 32 / 6500, 0.49% | 25 / 3500, 0.71% | 7 / 3000, 0.23% |
| whisper.cpp | 46 / 6500, 0.71% | 34 / 3500, 0.97% | 12 / 3000, 0.40% |
| Qwen3 Speech Recognition | 6 / 6500, 0.09% | 4 / 3500, 0.11% | 2 / 3000, 0.07% |
| ghost613 Korean Whisper | 0 / 6500, 0.00% | 0 / 3500, 0.00% | 0 / 3000, 0.00% |
| CrisperWhisper | 10 / 6500, 0.15% | 3 / 3500, 0.09% | 7 / 3000, 0.23% |
| Whisper-Streaming | 13 / 6500, 0.20% | 11 / 3500, 0.31% | 2 / 3000, 0.07% |
| SimulStreaming | 92 / 6391, 1.44% | 29 / 3461, 0.84% | 63 / 2930, 2.15% |

빈 출력, 비정상적으로 길거나 짧은 출력, 반복 패턴, 자막 제공류 문구도 관찰되었다. 비정상적인 길이는 reference 길이를 기준으로 판단했다.

| pattern | OpenAI | faster | whisper.cpp | Qwen3 Speech Recognition | ghost613 | CrisperWhisper | Whisper-Streaming | SimulStreaming |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 빈 출력 | 62, 0.95% | 61, 0.94% | 59, 0.91% | 1, 0.02% | 0, 0.00% | 20, 0.31% | 19, 0.29% | 2154, 33.70% |
| 긴 출력 | 16, 0.25% | 16, 0.25% | 0, 0.00% | 0, 0.00% | 182, 2.80% | 67, 1.03% | 10, 0.15% | 37, 0.58% |
| 짧은 출력 | 10, 0.15% | 9, 0.14% | 5, 0.08% | 0, 0.00% | 21, 0.32% | 47, 0.72% | 8, 0.12% | 1228, 19.21% |
| 반복 패턴 | 17, 0.26% | 17, 0.26% | 28, 0.43% | 11, 0.17% | 132, 2.03% | 164, 2.52% | 9, 0.14% | 48, 0.75% |
| "자막 제공"류 문구 | 11, 0.17% | 11, 0.17% | 0, 0.00% | 0, 0.00% | 0, 0.00% | 0, 0.00% | 4, 0.06% | 0, 0.00% |

SimulStreaming은 일부 샘플에서 decode 실패가 발생해 성공 prediction 기준으로 패턴 비율을 계산했다. Whisper-Streaming은 기존 Whisper 계열 decoder를 streaming 방식으로 감싸는 구조라 안정적으로 처리되었고, SimulStreaming은 현재 설정에서는 빈 출력과 짧은 출력이 많이 나타났다.
