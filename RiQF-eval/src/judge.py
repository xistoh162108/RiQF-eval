# src/judge.py (수정 완료된 전체 코드)

import logging
import json
import os
import re
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field

import google.generativeai as genai
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError

# --- 재시도(Retry) 라이브러리 임포트 ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from preprocess import PageContext
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------
# 0. 기본 설정 및 비동기 클라이언트 초기화
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
try:
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    async_openai_client = None
# (Gemini, Anthropic 클라이언트 초기화는 그대로 유지)
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except (TypeError, KeyError):
    logging.warning("GOOGLE_API_KEY가 설정되지 않아 Gemini 모델을 사용할 수 없습니다.")
try:
    async_anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except TypeError:
    logging.warning("ANTHROPIC_API_KEY가 설정되지 않아 Claude 모델을 사용할 수 없습니다.")
    async_anthropic_client = None

# --------------------------------------------------------------------------
# 1. 데이터 클래스 및 레지스트리 정의
# --------------------------------------------------------------------------
@dataclass
class JudgeScore:
    # 점수 타입을 float으로 변경하여 0.0 ~ 5.0 사이의 소수점 점수를 받을 수 있도록 함
    accuracy: float = 0.0
    groundedness: float = 0.0
    completeness: float = 0.0
    clarity: float = 0.0
    final_score: float = 0.0
    note: str = ""
    issues: List[str] = field(default_factory=list)

JUDGE_REGISTRY: Dict[str, Callable] = {}
def register_judge(name: str):
    def decorator(func: Callable): JUDGE_REGISTRY[name] = func; return func
    return decorator

# --------------------------------------------------------------------------
# 2. 휴리스틱 체크 함수
# --------------------------------------------------------------------------
def run_heuristic_checks(answer: str, context: PageContext, min_len: int = 10, max_len: int = 1000) -> Dict[str, Any]:
    """답변에 대한 간단한 휴리스틱 검사를 실행합니다."""
    checks = {
        "is_not_empty": bool(answer and answer.strip()),
        "is_within_length": min_len <= len(answer) <= max_len,
        "is_not_error_message": not answer.lower().startswith("오류:") and not answer.lower().startswith("error:"),
        "contains_korean": bool(re.search("[\uac00-\ud7a3]", answer))
    }
    return checks

# ========================================================================
# 3. LLM-as-a-Judge 구현
# ========================================================================
COMMON_RULES = """
**Evaluation Rules:**
1.  **Strictly Grounded:** You must evaluate the answer based *only* on the information present in the [Reference Context]. Do not use any external knowledge.
2.  **Be Objective:** Do not let the length or writing style of the answer influence your score. Focus only on the criteria below.
3.  **Output Format:** Your output MUST be a single, valid JSON object and nothing else.

**Evaluation Criteria (Score 0.0-5.0, with one decimal place for each):**
-   **Accuracy (weight: 0.4):** Is the information in the answer factually correct according to the context?
-   **Groundedness (weight: 0.35):** Is the answer fully supported and justified by the provided context?
-   **Completeness (weight: 0.15):** Does the answer address all parts of the user's question?
-   **Clarity (weight: 0.10):** Is the answer clear, concise, and easy to understand?

**Your JSON Output:**
Please provide your evaluation in the following JSON format. Calculate `final_score` as the weighted sum of the other scores (accuracy*0.4 + groundedness*0.35 + completeness*0.15 + clarity*0.10).

{{
  "accuracy": <0.0-5.0 float score>,
  "groundedness": <0.0-5.0 float score>,
  "completeness": <0.0-5.0 float score>,
  "clarity": <0.0-5.0 float score>,
  "final_score": <0.0-5.0 float score>,
  "note": "<Brief justification for your scores, in 30 words or less>",
  "issues": ["list", "of", "identified", "issues", "e.g.", "hallucination", "incorrect_number"]
}}
"""

# [변경점 1] 심사관 프롬프트 템플릿을 페르소나 중심으로 재구성합니다.
# 기존의 좋은 프롬프트들을 재사용하여 GPT-4o 기반의 페르소나 3개를 정의합니다.
JUDGE_PROMPT_TEMPLATES = {
    "gpt-4o_fact_checker": f"""
You are a **Forensic Fact-Checker**. 
Your primary and non-negotiable mission is to verify the **absolute factual accuracy and logical consistency** of the provided answer. Your evaluation should heavily focus on the **Accuracy** criterion.
{COMMON_RULES}
**Input:**
[Reference Context]
{{context_markdown}}
[Question]
{{question_text}}
[Answer to Evaluate]
{{answer_text}}
""",

    "gpt-4o_groundedness_analyst": f"""
You are a **Contextual Integrity Analyst**.
Your main responsibility is to ensure the answer is **fully supported by and comprehensively reflects the provided context**. Your evaluation should heavily focus on the **Groundedness** and **Completeness** criteria.
{COMMON_RULES}
**Input:**
[Reference Context]
{{context_markdown}}
[Question]
{{question_text}}
[Answer to Evaluate]
{{answer_text}}
""",

    "gpt-4o_clarity_specialist": f"""
You are a **Clarity and Readability Specialist**.
Your primary goal is to evaluate the answer from the perspective of a user who needs to quickly and easily understand the information. Your evaluation should heavily focus on the **Clarity** criterion.
{COMMON_RULES}
**Input:**
[Reference Context]
{{context_markdown}}
[Question]
{{question_text}}
[Answer to Evaluate]
{{answer_text}}
"""
}

RETRY_SETTINGS = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((OpenAIRateLimitError, AnthropicRateLimitError))
)

# [변경점 2] 여러 페르소나를 처리할 수 있는 단일 함수로 통합합니다.
# 데코레이터를 여러 번 사용하여, 정의된 모든 페르소나 이름을 이 함수에 등록합니다.
@RETRY_SETTINGS
@register_judge("gpt-4o_fact_checker")
@register_judge("gpt-4o_groundedness_analyst")
@register_judge("gpt-4o_clarity_specialist")
async def judge_with_gpt4o_persona(
    question: Dict, 
    context: PageContext, 
    answer: str,
    judge_model_name: str  # 어떤 페르소나를 사용할지 결정하는 이름
) -> JudgeScore:
    """GPT-4o를 기반으로 다양한 페르소나 심사를 수행하는 범용 함수"""
    if not async_openai_client: raise RuntimeError("OpenAI Async 클라이언트가 초기화되지 않았습니다.")
    
    # judge_model_name을 사용하여 해당 페르소나의 프롬프트를 동적으로 선택합니다.
    prompt_template = JUDGE_PROMPT_TEMPLATES.get(judge_model_name)
    if not prompt_template:
        raise ValueError(f"정의되지 않은 페르소나 심사관입니다: {judge_model_name}")

    prompt = prompt_template.format(
        context_markdown=context.markdown_content,
        question_text=question.get('question', ''),
        answer_text=answer
    )
    
    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o",  # 실제 API 호출은 모두 gpt-4o 모델을 사용합니다.
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_data = json.loads(response.choices[0].message.content)
        return JudgeScore(**response_data)
    except OpenAIRateLimitError as e:
        logging.warning(f"{judge_model_name} 심사 Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"{judge_model_name} 심사 최종 실패: {e}")
        return JudgeScore(note=f"Error during judging with {judge_model_name}: {e}")

# 참고: 다른 모델(Gemini, Claude) 기반의 심사관 함수는 이제 사용하지 않으므로 삭제하거나 주석 처리합니다.
# 필요하다면 남겨두고 main.py에서 --judges 인자로 호출할 수도 있습니다.

# ========================================================================
# 5. 쌍대 비교(Pairwise) 심사관 구현
# ========================================================================
PAIRWISE_PROMPT_TEMPLATE = """
You are a fair and impartial judge. Your task is to compare two answers (Answer A and Answer B) based on the user's question and the provided context. Choose which answer is better and briefly explain your reasoning.

**Evaluation Rules:**
1.  **Strictly Grounded:** Base your evaluation *only* on the information present in the [Reference Context].
2.  **Focus on Helpfulness:** Consider accuracy, completeness, and clarity. The better answer is the one that would be more helpful to the user.
3.  **Handle Ties:** If both answers are equally good or equally bad, you can declare a "tie".
4.  **Output Format:** Your output MUST be a single, valid JSON object and nothing else.

**Input:**
[Reference Context]
{context_markdown}

[Question]
{question_text}

[Answer A]
{answer_a}

[Answer B]
{answer_b}

**Your JSON Output:**
Choose the winner and provide a brief justification (under 30 words).

{{
  "winner": "<'A', 'B', or 'tie'>",
  "reason": "<Brief justification for your choice>"
}}
"""

@RETRY_SETTINGS
@register_judge("gpt-4o_pairwise_judge")
async def judge_pairwise_with_gpt4o(
    question: Dict,
    context: PageContext,
    answer: Dict[str, str], # answer는 {'A': '...', 'B': '...'} 형태
    judge_model_name: str
) -> Dict:
    """GPT-4o를 사용하여 두 답변을 비교 평가하는 심사관"""
    if not async_openai_client:
        raise RuntimeError("OpenAI Async 클라이언트가 초기화되지 않았습니다.")
    
    prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        context_markdown=context.markdown_content,
        question_text=question.get('question', ''),
        answer_a=answer.get('A', ''),
        answer_b=answer.get('B', '')
    )
    
    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_data = json.loads(response.choices[0].message.content)
        return response_data
    except Exception as e:
        logging.error(f"Pairwise judging 최종 실패: {e}")
        return {"winner": "error", "reason": str(e)}

# ========================================================================
# 4. 메인 심사 라우터 함수
# ========================================================================
async def judge_answer(
    judge_model_name: str,
    question: Dict[str, Any],
    context: PageContext,
    answer: str
) -> JudgeScore:
    """요청된 심사관을 찾아 채점을 위임하는 메인 함수"""
    if judge_model_name not in JUDGE_REGISTRY:
        raise ValueError(f"등록되지 않은 심사관 모델입니다: {judge_model_name}. 사용 가능: {list(JUDGE_REGISTRY.keys())}")
    
    logging.info(f"심사관 페르소나 '{judge_model_name}'을 사용하여 채점을 시작합니다.")
    judge_function = JUDGE_REGISTRY[judge_model_name]
    
    # [변경점 3] judge_function에 judge_model_name을 인자로 전달합니다.
    # 이것이 범용 심사 함수가 자신이 어떤 페르소나인지 알 수 있게 해주는 핵심 연결고리입니다.
    return await judge_function(question, context, answer, judge_model_name=judge_model_name)