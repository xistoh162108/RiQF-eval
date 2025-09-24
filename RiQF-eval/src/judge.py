# src/judge.py

import logging
import json
import os
import re
from typing import List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field

import google.generativeai as genai
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitErro



# --- 재시도(Retry) 라이브러리 임포트 (수정됨) ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

from preprocess import PageContext
from dotenv import load_dotenv
load_dotenv()  # .env 파일의 환경 변수를 os.environ에 로드

# --------------------------------------------------------------------------
# 0. 기본 설정 및 비동기 클라이언트 초기화
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
try:
    async_openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    async_openai_client = None
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except (TypeError, KeyError):
    logging.warning("GOOGLE_API_KEY가 설정되지 않아 Gemini 모델을 사용할 수 없습니다.")
try:
    async_anthropic_client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
except TypeError:
    logging.warning("ANTHROPIC_API_KEY가 설정되지 않아 Claude 모델을 사용할 수 없습니다.")
    async_anthropic_client = None

# ... (JudgeScore, JUDGE_REGISTRY, register_judge, run_heuristic_checks 등은 변경 없음) ...
@dataclass
class JudgeScore:
    accuracy: int = 0
    groundedness: int = 0
    completeness: int = 0
    clarity: int = 0
    final_score: float = 0.0
    note: str = ""
    issues: List[str] = field(default_factory=list)
JUDGE_REGISTRY: Dict[str, Callable] = {}
def register_judge(name: str):
    def decorator(func: Callable): JUDGE_REGISTRY[name] = func; return func
    return decorator
def run_heuristic_checks(answer: str, context: PageContext, min_len: int = 10, max_len: int = 1000) -> Dict[str, Any]:
    return {} # 내용 생략

# ========================================================================
# 3. LLM-as-a-Judge 구현 (재시도 데코레이터 수정)
# ========================================================================
# 범용 평가 규칙 및 JSON 출력 형식 정의 (기존과 동일)
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

# 심사관별 특화 프롬프트 딕셔너리
JUDGE_PROMPT_TEMPLATES = {
    "gpt-4o": f"""
You are a **Forensic Fact-Checker**. 
Your primary and non-negotiable mission is to verify the **absolute factual accuracy and logical consistency** of the provided answer.

**Evaluation Directives:**
-   **Accuracy Check:** Cross-reference every single fact, number, and statement in the answer against the [Reference Context]. If a fact is not explicitly stated or is slightly different (e.g., a number is off by one digit, a name is misspelled), the `accuracy` score must be penalized severely.
-   **Logical Coherence:** Assess if the answer's claims logically follow from the context. A score of 0.0 should be given for `accuracy` if the answer contains any form of hallucination, contradiction, or misleading information.
-   **Penalization:** You must not be lenient. Any single factual error, however small, should drop the `accuracy` score to 1.0 or 0.0.

{COMMON_RULES}

**Input:**

[Reference Context]
{{context_markdown}}

[Question]
{{question_text}}

[Answer to Evaluate]
{{answer_text}}
""",

    "gemini-1.5-pro": f"""
You are a **Contextual Integrity Analyst**.
Your main responsibility is to ensure the answer is **fully supported by and comprehensively reflects the provided context**.

**Evaluation Directives:**
-   **Groundedness Verification:** For every claim in the answer, you must find its direct source in the [Reference Context]. If a statement cannot be directly and explicitly traced back to the context, it is ungrounded. This must lead to a severe penalty on the `groundedness` score.
-   **Completeness Check:** Carefully compare the user's [Question] with the [Reference Context] and the [Answer to Evaluate]. Does the answer address all parts of the question that are answerable based on the context? If a key detail from the context that is relevant to the question is missing from the answer, the `completeness` score must be penalized.
-   **Penalization:** Any missing information or unsupported claim, however minor, should result in a low `completeness` or `groundedness` score.

{COMMON_RULES}

**Input:**

[Reference Context]
{{context_markdown}}

[Question]
{{question_text}}

[Answer to Evaluate]
{{answer_text}}
""",

    "claude-sonnet-4": f"""
You are a **Clarity and Readability Specialist**.
Your primary goal is to evaluate the answer from the perspective of a user who needs to quickly and easily understand the information.

**Evaluation Directives:**
-   **Clarity Assessment:** Rate how easily a non-expert user could understand the answer. Is the language simple? Is the structure logical?
-   **Simplicity and Directness:** An answer should get straight to the point. Penalize answers that are overly verbose, repetitive, or contain unnecessary jargon.
-   **Flow and Readability:** Check for a smooth reading experience. A correct answer that is poorly written, confusingly structured, or feels like a "wall of text" should receive a low `clarity` score. Focus on the user's ability to extract key information without effort.
-   **Penalization:** Even if an answer is factually correct, a low `clarity` score is warranted if it fails to be a good user experience.

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
    retry=retry_if_exception_type((OpenAIRateLimitErro, AnthropicRateLimitError))
)

@RETRY_SETTINGS
@register_judge("gpt-4o")
async def judge_openai_gpt4o(question: Dict, context: PageContext, answer: str) -> JudgeScore:
    if not async_openai_client: raise RuntimeError("OpenAI Async 클라이언트가 초기화되지 않았습니다.")
    prompt = JUDGE_PROMPT_TEMPLATES["gpt-4o"].format(
        context_markdown=context.markdown_content,
        question_text=question.get('question', ''),
        answer_text=answer
    )
    try:
        response = await async_openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        response_data = json.loads(response.choices[0].message.content)
        return JudgeScore(**response_data)
    except OpenAIRateLimitError as e:
        logging.warning(f"gpt-4o 심사 Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"gpt-4o 심사 최종 실패: {e}")
        return JudgeScore(note=f"Error during judging: {e}")

@RETRY_SETTINGS
@register_judge("gemini-1.5-pro")
async def judge_gemini_1_5_pro(question: Dict, context: PageContext, answer: str) -> JudgeScore:
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        prompt = JUDGE_PROMPT_TEMPLATES["gemini-1.5-pro"].format(
            context_markdown=context.markdown_content,
            question_text=question.get('question', ''),
            answer_text=answer
        )
        response = await model.generate_content_async(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        response_data = json.loads(response.text)
        return JudgeScore(**response_data)
    except Exception as e:
        logging.error(f"Gemini 1.5 Pro 심사 최종 실패: {e}")
        return JudgeScore(note=f"Error during judging: {e}")

@RETRY_SETTINGS
@register_judge("claude-sonnet-4")
async def judge_claude_sonnet_4(question: Dict, context: PageContext, answer: str) -> JudgeScore:
    if not async_anthropic_client: raise RuntimeError("Anthropic Async 클라이언트가 초기화되지 않았습니다.")
    base_prompt = JUDGE_PROMPT_TEMPLATES["claude-sonnet-4"]
    parts = base_prompt.split("**Input:**")
    system_prompt = parts[0]
    user_input_template = "**Input:**" + parts[1]
    prompt = user_input_template.format(
        context_markdown=context.markdown_content,
        question_text=question.get('question', ''),
        answer_text=answer
    )
    try:
        response = await async_anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024
        )
        json_string = response.content[0].text
        json_string = json_string[json_string.find('{'):json_string.rfind('}')+1]
        response_data = json.loads(json_string)
        return JudgeScore(**response_data)
    except AnthropicRateLimitError as e:
        logging.warning(f"Claude 심사 Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"Claude Sonnet 심사 최종 실패: {e}")
        return JudgeScore(note=f"Error during judging: {e}")

async def judge_answer(
    judge_model_name: str,
    question: Dict[str, Any],
    context: PageContext,
    answer: str
) -> JudgeScore:
    if judge_model_name not in JUDGE_REGISTRY:
        raise ValueError(f"등록되지 않은 심사관 모델입니다: {judge_model_name}.")
    
    logging.info(f"심사관 모델 '{judge_model_name}'을 사용하여 채점을 시작합니다.")
    judge_function = JUDGE_REGISTRY[judge_model_name]
    return await judge_function(question, context, answer)