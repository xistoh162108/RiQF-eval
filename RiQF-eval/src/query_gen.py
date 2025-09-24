# src/query_gen.py

import json
import logging
import os
from openai import AsyncOpenAI
from typing import List, Dict, Any
import argparse
import asyncio  # asyncio 임포트 추가

from preprocess import PageContext
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------
# 로깅 설정, 데이터 구조, OpenAI 클라이언트 초기화 (이전과 동일)
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
try:
    async_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    async_client = None

# ========================================================================
# SYSTEM_PROMPT (이전과 동일)
# ========================================================================
SYSTEM_PROMPT = """
You are an expert evaluator who generates high-quality evaluation questions based on a given document context.
Your goal is to create a comprehensive set of questions that fully test a model's understanding of the document.

Rules:
1.  Generate questions using *only* the information within the provided [CONTEXT]. The context is provided in Markdown format.
2.  Your output must be *only* the specified JSON object, without any other explanations.
3.  Generate a diverse range of questions with varying difficulty levels and types.

Question Types:
- `fact_extraction`: Extract a single, specific fact. (e.g., "What is the name of the author?")
- `list_extraction`: Extract a list of items or components. (e.g., "List the main steps of the experiment.")
- `table_lookup`: Find and extract a value from a table. (e.g., "What is the value in the second row and third column of Table 1?")
- `numeric_calculation`: Perform a simple calculation (e.g., addition, subtraction) based on numerical data in the context. (e.g., "What is the sum of the two numbers mentioned in the first paragraph?")
- `figure_comprehension`: Interpret a figure or image. (e.g., "Based on the chart, which country has the highest population?")
- `relationship_analysis`: Analyze relationships between different entities or concepts mentioned in the text. This is for more complex queries. (e.g., "Explain the relationship between the two variables shown in Figure 2.")
- `comparative_analysis`: Compare and contrast two or more items. This is for more complex queries. (e.g., "Compare the results of Experiment A and Experiment B based on the data provided.")
"""

# ========================================================================
# [변경] USER_PROMPT_TEMPLATE 수정: 질문 개수를 유동적으로 요청하도록 변경
# ========================================================================
USER_PROMPT_TEMPLATE = """
[CONTEXT]
{context_string}

[REQUEST]
Based on the context above, please generate between {min_questions} and {max_questions} high-quality evaluation questions.
The exact number should depend on the richness of the information in the context. **Prioritize quality over quantity.**
If the context is short or lacks diverse information, generating fewer, high-quality questions is better than forcing creation of poor ones.

Ensure a good mix of question types, including both simple and complex ones.
**Crucially, the value for the "question" field in the JSON output MUST be written in Korean.**

[OUTPUT FORMAT]
{{
  "questions": [
    {{
      "question": "첫 번째 '한국어' 질문을 작성하세요.",
      "type": "fact_extraction",
      "difficulty": "Easy"
    }},
    {{
      "question": "두 번째 '한국어' 질문 (예: 표 값 비교)을 작성하세요.",
      "type": "comparative_analysis",
      "difficulty": "Medium"
    }},
    // ... more questions if the context allows ...
  ]
}}
"""
# ========================================================================

# ========================================================================
# JSON 수정을 위한 Few-Shot 프롬프트 및 함수 (이전과 동일)
# ========================================================================
JSON_FIXER_PROMPT = """
You are a JSON fixer. Your task is to correct any syntax errors in the provided text to make it a valid JSON object. Do not add any new information or explanations. Only output the corrected JSON.

Here are some examples of how to fix broken JSON:

### Example 1: Trailing comma
[BROKEN JSON]
{
  "name": "John Doe",
  "age": 30,
}
[CORRECTED JSON]
{
  "name": "John Doe",
  "age": 30
}

### Example 2: Missing comma
[BROKEN JSON]
{
  "name": "Jane Doe"
  "age": 25
}
[CORRECTED JSON]
{
  "name": "Jane Doe",
  "age": 25
}

### Example 3: Text outside the JSON object
[BROKEN JSON]
Here is the JSON you requested:
{
  "questions": [{"question": "질문 내용"}]
}
I hope this helps!
[CORRECTED JSON]
{
  "questions": [{"question": "질문 내용"}]
}

Now, fix the following broken JSON text.

[BROKEN JSON]
{broken_json_string}
[CORRECTED JSON]
"""

async def fix_broken_json(broken_string: str, model: str = "gpt-3.5-turbo") -> str:
    """잘못된 형식의 JSON 문자열을 받아 Few-Shot 프롬프팅으로 수정합니다. (비동기)"""
    logging.warning("JSON 파싱 실패. 수정 모델을 호출하여 복구를 시도합니다.")
    if not async_client:
        raise RuntimeError("OpenAI async client is not initialized.")
        
    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JSON_FIXER_PROMPT.format(broken_json_string=broken_string)}],
            temperature=0.0,
        )
        fixed_json_string = response.choices[0].message.content
        logging.info("JSON 수정 모델이 응답을 반환했습니다.")
        return fixed_json_string
    except Exception as e:
        logging.error(f"JSON 수정 모델 호출 중 오류 발생: {e}")
        return broken_string

def _build_context_string(
    page_context: PageContext,
    include_captions: bool = True,
    max_caps: int = 8,
    max_cap_len: int = 280,
) -> str:
    """LLM에 전달할 전체 컨텍스트 문자열을 구성합니다."""
    
    # 1. OCR로 추출된 Markdown 텍스트 추가
    md = (page_context.markdown_content or "").strip()
    parts = [f"# OCR_AND_MARKDOWN\n{md}"] if md else []

    # [추가] 이미지 캡션 추가 로직 복원
    if include_captions and page_context.image_captions:
        # 캡션이 있는 경우에만 헤더 추가
        captions_part = ["# IMAGE_CAPTIONS"]
        
        # 캡션을 순회하며 컨텍스트에 추가
        count = 0
        for path, caption in page_context.image_captions.items():
            if count >= max_caps:
                logging.warning(f"최대 캡션 개수({max_caps})에 도달하여 일부 캡션을 제외합니다.")
                break
            
            file_name = Path(path).name
            # 캡션 길이 제한
            truncated_caption = caption[:max_cap_len]
            captions_part.append(f"- {file_name}: {truncated_caption}")
            count += 1
        
        # 생성된 캡션 문자열을 전체 파츠에 추가
        if len(captions_part) > 1: # 헤더 외에 실제 캡션 내용이 있을 경우
            parts.append("\n".join(captions_part))

    return "\n\n".join(parts).strip()


# ========================================================================
# [변경] generate_questions_for_page 함수 시그니처 및 로직 수정
# ========================================================================
async def generate_questions_for_page(
    page_context: PageContext,
    min_questions: int = 2,
    max_questions: int = 5,
    model: str = "gpt-4o-mini",
    include_captions: bool = True,
    max_caps: int = 8,
    max_cap_len: int = 280
) -> List[Dict[str, Any]]:
    """페이지 컨텍스트 기반으로 평가 질문을 생성합니다. (비동기)"""
    if not async_client:
        logging.error("OpenAI Async 클라이언트가 초기화되지 않아 질문 생성을 건너뜁니다.")
        return []

    logging.info(f"페이지 '{page_context.page_id}'에 대한 질문 생성을 시작합니다 (목표: {min_questions}~{max_questions}개).")
    
    context_text = _build_context_string(
        page_context,
        include_captions=include_captions,
        max_caps=max_caps,
        max_cap_len=max_cap_len,
    )

    if len(context_text.split()) < 10:
        logging.warning(f"페이지 '{page_context.page_id}'의 컨텍스트가 너무 짧아 질문 생성을 건너뜁니다.")
        return []

    # [변경] 프롬프트 포맷팅에 min_questions, max_questions 사용
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context_string=context_text, 
        min_questions=min_questions,
        max_questions=max_questions
    )

    try:
        response = await async_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        response_content = response.choices[0].message.content
        try:
            generated_data = json.loads(response_content)
            return generated_data.get("questions", [])
        except json.JSONDecodeError:
            corrected_json_string = await fix_broken_json(response_content)
            generated_data = json.loads(corrected_json_string)
            return generated_data.get("questions", [])
    except Exception as e:
        logging.error(f"API 호출 또는 JSON 파싱 중 오류 발생: {e}")
        return []

# --------------------------------------------------------------------------
# [변경] 스크립트 실행 테스트 로직 수정
# --------------------------------------------------------------------------
async def test_run(args):
    """테스트를 위한 비동기 실행 함수"""
    logging.info(f"query_gen.py 모듈 테스트 시작 (입력 파일: {args.input_json})")
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        reconstructed_context = PageContext(**data)
        logging.info(f"'{reconstructed_context.page_id}' 페이지 컨텍스트를 성공적으로 로드했습니다.")
        
        # [변경] min/max 인자를 사용하여 함수 호출
        generated_questions = await generate_questions_for_page(
            reconstructed_context, 
            min_questions=args.min_questions,
            max_questions=args.max_questions
        )

        if generated_questions:
            print("\n" + "="*50)
            print(f"[생성된 질문 목록 ({len(generated_questions)}개)]")
            from pprint import pprint
            pprint(generated_questions)
            print("="*50)
        else:
            logging.warning("테스트 실행에서 질문이 생성되지 않았습니다.")
            
    except Exception as e:
        logging.error(f"테스트 중 오류 발생: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="전처리된 JSON 파일을 이용해 질문 생성 모듈을 테스트합니다.")
    parser.add_argument("--input_json", type=str, required=True, help="main.py가 생성한 PageContext JSON 파일의 경로.")
    # [변경] 커맨드 라인 인자 추가
    parser.add_argument("--min-questions", type=int, default=2, help="생성할 최소 질문 개수.")
    parser.add_argument("--max-questions", type=int, default=5, help="생성할 최대 질문 개수.")
    
    args = parser.parse_args()
    
    asyncio.run(test_run(args))