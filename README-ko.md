# RiQF-eval: 멀티모달 문서 QA 시스템을 위한 엄밀한 정량적 평가 프레임워크

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![KAIST](https://img.shields.io/badge/KAIST-School%20of%20Computing-blue)](https://cs.kaist.ac.kr/)
[![HyperX](https://img.shields.io/badge/HyperX-AI-purple)](https://ax.hyper-x.ai/)

**다른 언어로 읽기:** [**English (영어)**](./README.md)

---

**RiQF-eval**은 문서 파싱, 질문 생성, 그리고 비전-언어 모델(VLM)의 엄격한 평가를 위한 포괄적인 시스템입니다. 이 저장소는 멀티모달(multimodal) 문서 질의응답(QA) 시스템의 정량적 평가를 위한 완전하고 재현 가능한 방법론 프레임워크를 제시합니다.

이 프로젝트는 **(주)하이퍼엑스**와 **KAIST 린스타트업(Lean Startup) 프로그램**의 협력 프로젝트입니다.

## 초록 (Abstract)

본 프로젝트는 멀티모달 문서 QA 시스템의 정량적 평가를 위한 포괄적이고 재현 가능한 프레임워크를 소개합니다. 이 프로토콜은 완전 자동화된 파이프라인을 통해 다양한 입력 양식(modality) 조합에 따른 시스템 성능을 엄밀하게 평가하도록 설계되었습니다. 이 파이프라인은 (1) 표준화된 코퍼스 전처리 및 테스트 케이스 생성, (2) 공정한 조건 하의 통제된 답변 생성, 그리고 (3) 다수의 대규모 언어 모델을 심사관으로 활용하는 새로운 다단계 앙상블 평가 프로토콜(LLM-as-a-Judge)을 포함합니다. 단일 심사관의 편향을 완화하고 평가 안정성을 향상시키기 위해 점수 정규화, 신뢰도 기반 가중 집계, 평가자 간 신뢰도(IAA) 측정을 위한 공식적인 절차를 도입합니다. 성능은 품질, 지연 시간, 계산 비용의 세 가지 차원에서 정량화되며, 시스템의 상충 관계(trade-off)에 대한 총체적 이해를 위해 파레토 최적 전선(Pareto frontier) 분석을 가능하게 합니다.

## 핵심 방법론

저희 프레임워크는 검증 가능하고 일반화 가능한 과학적 주장을 도출하기 위해, 엄격한 학술 수준의 방법론에 기반하여 구축되었습니다.

### 1. 문제 정의

주요 목표는 멀티모달 QA 시스템(함수 $f$로 지칭)을 평가하는 것입니다. 이 시스템은 질의(query) $q$와 문서 페이지의 컨텍스트 입력이 주어졌을 때 답변 $a$를 생성합니다.

-   **입력**: 이미지 $I$, OCR 텍스트 $T$, 그리고 시각 요소에 대한 캡션 $C$로 표현되는 문서 페이지.
-   **질의**: 페이지 $p$에 대해 자동 생성된 질문 집합 $Q_p = \{q_1, ..., q_N\}$ 중 질문 $q_i$.
-   **평가 대상 시스템 (SUT)**: $M$개의 서로 다른 입력 양식 조합 $C^{(m)}$ 하에서 평가되는 함수 $f$.
-   **출력**: 평가의 대상이 되는 답변 $a_i^{(m)} = f(q_i, C^{(m)})$.

### 2. 입력 양식 조합

각 입력 양식의 기여도를 체계적으로 분석하기 위해, 애블레이션 연구(ablation studies)를 위한 고정된 조합 세트를 정의합니다:

-   **M1**: 이미지 단독: $C^{(1)} = \{I\}$
-   **M2**: 이미지 + 캡션: $C^{(2)} = \{I, C\}$
-   **M3**: 이미지 + OCR 텍스트: $C^{(3)} = \{I, T\}$
-   **M4**: 이미지 + 캡션 + OCR 텍스트: $C^{(4)} = \{I, C, T\}$
-   **M5**: 잘라낸 영역 + 캡션 + OCR 텍스트: $C^{(5)} = \{\{I_r\}, C, T\}$
-   **M6**: 텍스트만 (OCR + 캡션): $C^{(6)} = \{T, C\}$
-   **M7**: 텍스트만 (OCR): $C^{(7)} = \{T\}$

### 3. 자동화된 테스트 케이스 생성 (`QueryGen`)

전용 언어 모델(`LLM #1`)이 표준화되고 확장 가능한 테스트 세트를 생성하며, 이미지 픽셀이나 정답 정보로부터 독립성을 유지하기 위해 텍스트 컨텍스트 $\{T, C\}$에만 의존하여 작동합니다.

### 4. 다단계 평가 프로토콜

생성된 각 답변은 2단계 평가를 거칩니다:

1.  **1단계: 휴리스틱 필터링**: 경량의 규칙 기반 절차를 통해 **근거 유효성(Grounding Validity)**, **수치 일관성(Numerical Consistency)**, **형식 준수(Format Compliance)**를 검사하여 초기 품질 신호를 제공합니다.

2.  **2단계: 다중 심사관 LLM 앙상블 평가**: 서로 다른 LLM 패널(예: GPT-4o, Gemini 1.5 Pro, Claude 3 Sonnet)이 각 답변을 독립적으로 평가합니다. 편향을 완화하기 위해 시스템의 신원은 익명으로 처리되며, 모든 심사관은 표준화된 루브릭을 사용하여 **정확성(Accuracy)**, **근거성(Groundedness)**, **완전성(Completeness)**, **명료성(Clarity)**을 기준으로 답변을 채점합니다.

    심사관 패널의 점수들은 로버스트 z-점수 정규화(robust z-score normalization)를 거친 후, **신뢰도 가중 앙상블(reliability-weighted ensemble)** 방식을 통해 단일의 안정적인 최종 점수 $S^{\star}$로 집계됩니다.

### 5. 정량적 지표 및 분석

세 가지 범주에 걸쳐 지표를 수집합니다:

1.  **품질**: 최종 앙상블 점수 $S^{\star}$.
2.  **지연 시간 (Latency)**: 답변 생성에 소요된 시간(ms).
3.  **비용 (Cost)**: 질의당 계산 비용(USD).

통계적 유의성은 다중 비교를 위해 **홀름-본페로니 교정(Holm-Bonferroni correction)**을 적용한 **윌콕슨 부호 순위 검정(Wilcoxon signed-rank test)**을 사용하여 결정됩니다. 결과는 품질과 효율성 간의 상충 관계를 시각화하기 위해 **파레토 최적 전선(Pareto frontier) 그래프**로 종합됩니다.



---

## 주요 기능

-   **고급 PDF 처리**: MinerU를 사용하여 레이아웃 감지 및 콘텐츠 추출.
-   **멀티모달 분석**: 텍스트, 레이아웃, 이미지 처리를 결합하여 포괄적인 문서 이해.
-   **자동 질문 생성**: 문서 내용으로부터 구조화되고 다양한 질문 생성.
-   **통제된 답변 생성**: 다양한 VLM 모델과 여러 조합 전략을 사용하여 답변 생성.
-   **강건한 평가 파이프라인**: 다중 심사관 앙상블을 활용한 휴리스틱 및 모델 기반 평가로 답변 품질 심사.
-   **비동기 처리**: 정교한 재시도 메커니즘으로 대규모 문서 처리를 효율적으로 처리.

## 프로젝트 구조

```

RiQF-eval/
├── .env                  \# 환경 변수 (API 키)
├── README.md             \# 영문 README 파일
├── README-ko.md          \# 본 파일 (한글 README)
├── requirements.txt      \# 프로젝트 의존성
├── data/                 \# 입력 문서 (예: test\_1.pdf)
├── notebooks/            \# 분석용 Jupyter 노트북
├── results/              \# 모든 처리 단계의 출력 디렉토리
│   └── [document\_name]/
│       ├── pages/
│       ├── mineru\_raw\_output/
│       ├── preprocessed\_context/
│       ├── generated\_questions/
│       └── evaluation\_results/
└── src/                  \# 소스 코드
├── main.py           \# 메인 실행 스크립트
├── preprocess.py     \# MinerU를 이용한 문서 전처리
├── query\_gen.py      \# 질문 생성 모듈
├── answer\_gen.py     \# 답변 생성 모듈
├── judge.py          \# 답변 평가 및 심사 모듈
└── analysis.py       \# 결과 분석 및 지표 계산

````

## 설치

### 사전 요구사항

1.  **Python 환경**: Python 3.8 이상
2.  **MinerU**: 문서 파싱 엔진.
3.  **Poppler**: PDF-이미지 변환을 위한 PDF 렌더링 라이브러리.
4.  **API 키**: 모델 접근을 위한 OpenAI, Anthropic, 또는 Google AI API 키.

### 설정 방법

1.  **의존성 설치**:
    ```bash
    # uv 패키지 매니저 설치
    pip install uv

    # MinerU 핵심 의존성 설치
    uv pip install "mineru[core]"

    # 추가 요구사항 설치
    pip install -r requirements.txt
    ```

2.  **시스템 의존성 설치**:
    ```bash
    # macOS (Homebrew 사용)
    brew install poppler

    # Ubuntu/Debian
    sudo apt-get install -y poppler-utils
    ```

3.  **환경 설정**: 프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 API 키를 추가합니다.
    ```bash
    # API 키를 포함한 .env 파일 생성
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    ```

## 사용법

### 커맨드 라인 인터페이스

메인 스크립트는 다양한 플래그를 통해 포괄적인 문서 처리를 지원합니다:

```bash
python src/main.py [OPTIONS]
````

### 커맨드 라인 인자

| 인자                   | 타입  | 기본값                                           | 설명                                             |
| ---------------------- | ----- | ------------------------------------------------ | ------------------------------------------------ |
| `--input`              | str   | `"data/"`                                        | 처리할 파일 또는 폴더의 경로.                    |
| `--output`             | str   | `"results/"`                                     | 모든 결과가 저장될 출력 디렉토리.                |
| `--concurrency`        | int   | `5`                                              | 최대 동시 비동기 작업 수.                        |
| `--interval`           | float | `1.5`                                            | API 요청 간 최소 대기 시간 (초).                 |
| `--save_context`       | flag  | `False`                                          | 전처리된 페이지 컨텍스트를 JSON 파일로 저장.     |
| `--captioning`         | flag  | `False`                                          | VLM을 사용한 이미지 캡셔닝 활성화.               |
| `--generate_questions` | flag  | `False`                                          | 문서 내용으로부터 질문 생성.                     |
| `--generate_answers`   | flag  | `False`                                          | 생성된 질문에 대한 답변 생성.                    |
| `--judge_answers`      | flag  | `False`                                          | 생성된 답변의 품질 평가.                         |
| `--models`             | list  | `["gpt-4o"]`                                     | 답변 생성에 사용할 모델.                         |
| `--combos`             | list  | `["M1", "M2", ...]`                              | 테스트할 입력 양식 조합 전략.                    |
| `--judges`             | list  | `["gpt-4o", "gemini-1.5-pro", "claude-3-sonnet"]`| 평가 앙상블에 사용할 심사관 모델.                |
| `--run_analysis`       | flag  | `False`                                          | 최종 분석 및 보고 단계 실행.                     |

### 사용 예시

#### 단일 문서에 전체 파이프라인 실행

```bash
python src/main.py \
    --input data/test_1.pdf \
    --output results/test_run \
    --save_context \
    --captioning \
    --generate_questions \
    --generate_answers \
    --judge_answers \
    --run_analysis
```

#### 사용자 정의 모델로 배치 처리

```bash
python src/main.py \
    --input data/ \
    --output results/batch_custom \
    --concurrency 4 \
    --interval 2.0 \
    --models gpt-4-turbo gpt-4o \
    --combos M1 M4 M7 \
    --judges gpt-4o \
    --generate_answers \
    --judge_answers
```

## 성능 최적화

이 시스템은 대규모 처리 및 API 제한을 원활하게 처리하기 위해 정교한 속도 제한 및 재시도 메커니즘을 포함합니다.

  - **속도 제한 (Rate Limiting)**: 설정 가능한 동시성 수준 및 요청 간 간격 기반 제어.
  - **재시도 메커니즘 (Retry Mechanism)**: 실패한 작업에 대해 최대 5회까지 지수적 백오프(exponential backoff)를 구현하며, API 속도 제한을 피하기 위한 지능적인 대기 시간 적용.
  - **메모리 관리 (Memory Management)**: 대용량 문서 및 결과 집합을 효율적으로 처리하기 위해 비동기 처리 및 스트리밍 JSONL 출력 사용.

## 문제 해결 (Troubleshooting)

1.  **속도 제한 오류**: 동시성을 줄이고 간격을 늘리세요.
    `--concurrency 2 --interval 3.0`
2.  **API 키 문제**: `.env` 파일이 올바른 형식으로 프로젝트 루트에 위치해 있는지 확인하세요.
3.  **메모리 문제**: 매우 큰 문서의 경우, 더 낮은 동시성으로 처리하세요.
    `--concurrency 1`

## 인용 (Citation)

이 프레임워크를 연구에 사용하시는 경우, 아래 저자들을 인용해 주시기 바랍니다:

  - **박지민**: xistoh162108@kaist.ac.kr, xistoh162108@hyper-x.ai
      - (주)하이퍼엑스 개발 인턴
      - KAIST 전산학부 학부생
  - **장종례**: cto@hyper-x.ai
      - (주)하이퍼엑스 최고기술책임자 (CTO)
  - **지한빈**: ceo@hyper-x.ai
      - (주)하이퍼엑스 최고경영자 (CEO)

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.