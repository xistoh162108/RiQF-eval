# src/answer_gen.py

import logging
import os
import base64
from typing import List, Dict, Any, Callable, Tuple

# --- 비동기 클라이언트 및 관련 라이브러리 임포트 ---
import google.generativeai as genai
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from PIL import Image
import torch
# --- Hugging Face 로컬 모델 라이브러리 임포트 ---
# --- Hugging Face 로컬 모델 라이브러리 임포트 ---
# --- Hugging Face 로컬 모델 라이브러리 임포트 ---
# --- Hugging Face 로컬 모델 라이브러리 임포트 ---
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    MllamaForConditionalGeneration, 
    AutoProcessor as MllamaAutoProcessor,
    PaliGemmaProcessor, 
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoProcessor as SmolVLMAutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor as QwenAutoProcessor,
    # Idefics3를 위한 클래스 추가
    AutoModelForVision2Seq,
    AutoProcessor as IdeficsAutoProcessor,
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info

# --- DeepSeek-VL 전용 라이브러리 임포트 ---
from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images


# --- 재시도(Retry) 라이브러리 임포트 (수정됨) ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# --- 다른 모듈 임포트 ---
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
    logging.warning("OPENAI_API_KEY가 설정되지 않아 OpenAI 모델을 사용할 수 없습니다.")
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

# --- 로컬 Hugging Face 모델 초기화 ---
# 스크립트 시작 시 한 번만 실행됩니다.
# 1. 사용 가능한 장치를 우선순위에 따라 결정
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"선택된 장치(Device): {DEVICE}")

# 2. 장치에 따라 최적의 데이터 타입(Dtype) 설정
DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16
logging.info(f"선택된 데이터 타입(Dtype): {DTYPE}")

# Llama-3.2-Vision 모델 로딩
try:
    logging.info("Llama-3.2-Vision 모델 로딩을 시작합니다...")
    llama_model_id = "meta-llama/Llama-3.2-11B-Vision"
    llama_model = MllamaForConditionalGeneration.from_pretrained(
        llama_model_id, torch_dtype=DTYPE, device_map="auto"
    ).eval()
    llama_processor = AutoProcessor.from_pretrained(llama_model_id)
    logging.info("Llama-3.2-Vision 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"Llama-3.2-Vision 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    llama_model, llama_processor = None, None

# PaliGemma 모델 로딩
try:
    logging.info("PaliGemma 모델 로딩을 시작합니다...")
    paligemma_model_id = "google/paligemma2-3b-pt-896"
    paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
        paligemma_model_id, dtype=DTYPE, device_map="auto"
    ).eval()
    paligemma_processor = PaliGemmaProcessor.from_pretrained(paligemma_model_id)
    # PaliGemma 전용 이미지 프로세서 설정
    paligemma_processor.image_processor.do_resize = True
    paligemma_processor.image_processor.size = {"height": 896, "width": 896}
    logging.info("PaliGemma 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"PaliGemma 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    paligemma_model, paligemma_processor = None, None
# SmolVLM 모델 로딩
try:
    logging.info("SmolVLM 모델 로딩을 시작합니다...")
    smolvlm_model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    smolvlm_model = AutoModelForImageTextToText.from_pretrained(
        smolvlm_model_id, dtype=DTYPE, device_map="auto"
    ).eval()
    smolvlm_processor = AutoProcessor.from_pretrained(smolvlm_model_id)
    logging.info("SmolVLM 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"SmolVLM 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    smolvlm_model, smolvlm_processor = None, None
    
# InternVL 모델 로딩
try:
    logging.info("InternVL 모델 로딩을 시작합니다...")
    internvl_model_id = "OpenGVLab/InternVL2_5-8B"
    internvl_model = AutoModel.from_pretrained(
        internvl_model_id,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(DEVICE).eval()
    internvl_tokenizer = AutoTokenizer.from_pretrained(internvl_model_id, trust_remote_code=True, use_fast=False)
    logging.info("InternVL 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"InternVL 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    internvl_model, internvl_tokenizer = None, None
    
# Qwen2.5-VL 모델 로딩
try:
    logging.info("Qwen2.5-VL 모델 로딩을 시작합니다...")
    qwen_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    qwen_processor = QwenAutoProcessor.from_pretrained(qwen_model_id)
    logging.info("Qwen2.5-VL 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"Qwen2.5-VL 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    qwen_model, qwen_processor = None, None

# Idefics3 모델 로딩
try:
    logging.info("Idefics3 모델 로딩을 시작합니다...")
    idefics_model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
    idefics_model = AutoModelForVision2Seq.from_pretrained(
        idefics_model_id, torch_dtype=DTYPE, device_map="auto"
    ).eval()
    idefics_processor = IdeficsAutoProcessor.from_pretrained(idefics_model_id)
    logging.info("Idefics3 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"Idefics3 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    idefics_model, idefics_processor = None, None

# DeepSeek-VL2 모델 로딩
try:
    logging.info("DeepSeek-VL2 모델 로딩을 시작합니다...")
    deepseek_model_id = "deepseek-ai/deepseek-vl2-small"
    deepseek_processor = DeepseekVLV2Processor.from_pretrained(deepseek_model_id)
    deepseek_model = AutoModelForCausalLM.from_pretrained(
        deepseek_model_id, trust_remote_code=True
    ).to(DTYPE).to(DEVICE).eval()
    logging.info("DeepSeek-VL2 모델 로딩 완료.")
except Exception as e:
    logging.warning(f"DeepSeek-VL2 모델 로딩 실패: {e}. 해당 모델을 사용할 수 없습니다.")
    deepseek_model, deepseek_processor = None, None

# --------------------------------------------------------------------------
# 1. 모델 레지스트리 및 데코레이터 (변경 없음)
# --------------------------------------------------------------------------
VLM_REGISTRY: Dict[str, Callable] = {}
def register_vlm(name: str):
    def decorator(func: Callable): VLM_REGISTRY[name] = func; return func
    return decorator

# ========================================================================
# 2. 모델별 답변 생성 함수 구현 (재시도 데코레이터 수정)
# ========================================================================

RETRY_SETTINGS = retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((OpenAIRateLimitError, AnthropicRateLimitError))
)

@RETRY_SETTINGS
@register_vlm("gpt-4o")
async def generate_answer_openai_gpt4o(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    if not async_openai_client: raise RuntimeError("OpenAI Async 클라이언트가 초기화되지 않았습니다.")
    system_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "**Your final answer MUST be written in Korean.**\n"
        "**Crucially, for each key piece of information in your answer, briefly state your reasoning or the source from the context in parentheses `()`.**"
    )
    user_content = []
    user_content.append({"type": "text", "text": f"Question: {question['question']}"})
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_content.append({"type": "text", "text": f"\n[Text Context]\n{context.markdown_content}"})
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}})
        except FileNotFoundError: logging.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
    try:
        response = await async_openai_client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=1024, temperature=0.1)
        answer = response.choices[0].message.content
        token_usage = response.usage.model_dump()
        return answer, token_usage
    except OpenAIRateLimitError as e:
        logging.warning(f"gpt-4o Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"gpt-4o API 최종 호출 실패: {e}")
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

@RETRY_SETTINGS
@register_vlm("gemini-1.5-pro")
async def generate_answer_gemini_1_5_pro(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction="...")
        prompt_parts = []
        prompt_parts.append(f"Question: {question['question']}")
        if model_combo in ["M3", "M4", "M5", "M6", "M7"]: prompt_parts.append(f"\n[Text Context]\n{context.markdown_content}")
        image_paths = []
        if model_combo in ["M1", "M2", "M3", "M4"]:
            if context.source_image_path: image_paths.append(context.source_image_path)
        elif model_combo == "M5":
            if context.image_paths: image_paths.extend(context.image_paths)
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                prompt_parts.append(img)
            except FileNotFoundError: logging.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")
        response = await model.generate_content_async(prompt_parts)
        answer = response.text
        token_usage = {"prompt_tokens": response.usage_metadata.prompt_token_count, "completion_tokens": response.usage_metadata.candidates_token_count, "total_tokens": response.usage_metadata.total_token_count}
        return answer, token_usage
    except Exception as e:
        logging.error(f"Gemini 1.5 Pro API 최종 호출 실패: {e}")
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

@RETRY_SETTINGS
@register_vlm("claude-sonnet-4")
async def generate_answer_claude_sonnet_4(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    if not async_anthropic_client: raise RuntimeError("Anthropic Async 클라이언트가 초기화되지 않았습니다.")
    system_prompt = "..."
    messages = []
    content_block = []
    content_block.append({"type": "text", "text": f"Question: {question['question']}"})
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: content_block.append({"type": "text", "text": f"\n[Text Context]\n{context.markdown_content}"})
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                content_block.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64_image}})
        except FileNotFoundError: logging.warning(f"이미지 파일을 찾을 수 없습니다: {img_path}")
    messages.append({"role": "user", "content": content_block})
    try:
        response = await async_anthropic_client.messages.create(model="claude-sonnet-4-20250514", system=system_prompt, messages=messages, max_tokens=1024, temperature=0.1)
        answer = response.content[0].text
        token_usage = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens, "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
        return answer, token_usage
    except AnthropicRateLimitError as e:
        logging.warning(f"Claude Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"Claude Sonnet 4 API 최종 호출 실패: {e}")
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
# --- 새로 추가된 Llama-3.2-Vision 답변 생성 함수 ---
def _run_llama_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """Llama 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 프롬프트 구성
    user_prompt = f"Question: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"
    
    bos = llama_processor.tokenizer.bos_token or ""
    image_token = "<|image|>"
    # System prompt와 유사한 역할을 프롬프트 상단에 추가
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    prompt = f"{bos}{image_token}\n{system_like_prompt}\n\n{user_prompt}"

    # 2. 이미지 로드
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    
    # Llama는 단일 이미지만 처리하므로, 여러 개일 경우 첫 번째 이미지만 사용
    image_to_process = None
    if image_paths:
        try:
            image_to_process = Image.open(image_paths[0]).convert("RGB")
        except FileNotFoundError:
            logging.warning(f"Llama 모델을 위한 이미지 파일을 찾을 수 없습니다: {image_paths[0]}")
    
    # 3. 입력 데이터 처리
    inputs = llama_processor(images=image_to_process, text=prompt, return_tensors="pt").to(llama_model.device)

    # 4. 텍스트 생성 (모델 추론)
    eos_id = llama_processor.tokenizer.eos_token_id
    output_ids = llama_model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id
    )
    
    # 5. 결과 디코딩
    gen_only_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    generated_text = llama_processor.batch_decode(
        gen_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0].strip()

    # 토큰 사용량은 로컬 모델에서 직접 제공되지 않으므로, 입력/출력 텍스트 길이로 추정
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(gen_only_ids[0])
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return generated_text, token_usage


@register_vlm("llama-3.2-vision")
async def generate_answer_llama3_2_vision(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 Llama-3.2-Vision 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not llama_model or not llama_processor:
        raise RuntimeError("Llama-3.2-Vision 모델이 초기화되지 않았습니다.")

    try:
        # 동기적인 모델 추론 코드를 별도의 스레드에서 실행하여 이벤트 루프를 막지 않음
        answer, token_usage = await asyncio.to_thread(
            _run_llama_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"Llama-3.2-Vision 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
def _run_paligemma_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """PaliGemma 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 프롬프트 구성
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    user_prompt = f"Question: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"
    
    # PaliGemma 프롬프트 형식: <image>\nanswer ko {질문}
    prompt = f"<image>\nanswer ko {system_like_prompt}\n\n{user_prompt}"

    # 2. 이미지 로드
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    
    image_to_process = None
    if image_paths:
        try:
            image_to_process = Image.open(image_paths[0]).convert("RGB")
        except FileNotFoundError:
            logging.warning(f"PaliGemma 모델을 위한 이미지 파일을 찾을 수 없습니다: {image_paths[0]}")

    # 3. 입력 데이터 처리
    inputs = paligemma_processor(images=image_to_process, text=prompt, return_tensors="pt").to(DEVICE)

    # 4. 텍스트 생성 (모델 추론)
    with torch.no_grad():
        generated_ids = paligemma_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # 5. 결과 디코딩 (프롬프트 부분 제외)
    # PaliGemma는 프롬프트를 포함하여 생성하므로, 프롬프트 부분을 잘라냅니다.
    output_str = paligemma_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    prompt_part = f"answer ko {system_like_prompt}\n\n{user_prompt}" # <image> 제외
    generated_text = output_str.replace(prompt_part, "").strip()
    
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids[0]) - prompt_tokens
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    
    return generated_text, token_usage


@register_vlm("paligemma2")
async def generate_answer_paligemma(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 PaliGemma 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not paligemma_model or not paligemma_processor:
        raise RuntimeError("PaliGemma 모델이 초기화되지 않았습니다.")
    
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_paligemma_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"PaliGemma 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    

# --- 새로 추가된 SmolVLM 답변 생성 함수 ---
def _run_smolvlm_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """SmolVLM 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 대화형 프롬프트 구성
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"
    
    # SmolVLM의 content는 이미지와 텍스트의 리스트로 구성
    content = [{"type": "text", "text": user_prompt_text}]
    
    # 2. 이미지 경로 추가
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)

    # content 리스트의 맨 앞에 이미지들을 추가
    for img_path in reversed(image_paths): # reversed()로 순서 유지
        content.insert(0, {"type": "image", "path": img_path})

    conversation = [{"role": "user", "content": content}]

    # 3. 입력 데이터 처리 (apply_chat_template 사용)
    inputs = smolvlm_processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(smolvlm_model.device)
    
    # 4. 텍스트 생성 (모델 추론)
    with torch.no_grad():
        output_ids = smolvlm_model.generate(**inputs, max_new_tokens=1024, do_sample=False)

    # 5. 결과 디코딩
    generated_text = smolvlm_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    # 생성된 텍스트에서 프롬프트 부분을 제거 (chat template의 응답 형식에 따라)
    # 보통 'assistant\n' 뒤에 답변이 나옴
    answer_part = generated_text.split("assistant\n")
    final_answer = answer_part[1].strip() if len(answer_part) > 1 else generated_text.strip()

    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(output_ids[0]) - prompt_tokens
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return final_answer, token_usage

@register_vlm("smolvlm")
async def generate_answer_smolvlm(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 SmolVLM 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not smolvlm_model or not smolvlm_processor:
        raise RuntimeError("SmolVLM 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_smolvlm_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"SmolVLM 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
def _run_internvl_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """InternVL 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 이미지 로드 및 전처리
    pixel_values = None
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    
    # InternVL은 여러 이미지를 torch.cat으로 합쳐서 처리 가능
    if image_paths:
        try:
            # load_image는 InternVL 전용 헬퍼 함수
            pixel_values_list = [load_image(p, max_num=12).to(DTYPE).to(DEVICE) for p in image_paths]
            pixel_values = torch.cat(pixel_values_list, dim=0)
        except FileNotFoundError:
            logging.warning(f"InternVL 모델을 위한 이미지 파일을 찾을 수 없습니다.")

    # 2. 프롬프트 구성
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    # InternVL은 프롬프트에 이미지 개수만큼 <image> 토큰을 포함해야 할 수 있음
    image_tokens = '\n'.join(['<image>' for _ in image_paths])
    
    user_prompt = f"{image_tokens}\n{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"

    # 3. 텍스트 생성 (모델 추론)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    
    with torch.no_grad():
        response = internvl_model.chat(
            tokenizer=internvl_tokenizer,
            pixel_values=pixel_values,
            question=user_prompt,
            generation_config=generation_config
        )

    # 토큰 사용량 추정 (InternVL은 직접 제공하지 않음)
    prompt_tokens = len(internvl_tokenizer.encode(user_prompt))
    completion_tokens = len(internvl_tokenizer.encode(response))
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return response, token_usage

@register_vlm("internvl-8b")
async def generate_answer_internvl(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 InternVL 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not internvl_model or not internvl_tokenizer:
        raise RuntimeError("InternVL 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_internvl_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"InternVL 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
# --- 새로 추가된 Qwen2.5-VL 답변 생성 함수 ---
def _run_qwen_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """Qwen2.5-VL 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 대화형 프롬프트 구성
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"
        
    content = [{"type": "text", "text": user_prompt_text}]
    
    # 2. 이미지 경로 추가
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)

    for img_path in reversed(image_paths):
        # Qwen 유틸리티는 'file://' 접두사를 사용할 수 있음
        content.insert(0, {"type": "image", "image": f"file://{img_path}"})
        
    messages = [{"role": "user", "content": content}]

    # 3. 입력 데이터 처리 (qwen_vl_utils 사용)
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = qwen_processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(qwen_model.device)

    # 4. 텍스트 생성 (모델 추론)
    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=1024)

    # 5. 결과 디코딩 (프롬프트 부분 제외)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    final_answer = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids_trimmed[0])
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return final_answer, token_usage


@register_vlm("qwen2.5-vl")
async def generate_answer_qwen(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 Qwen2.5-VL 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not qwen_model or not qwen_processor:
        raise RuntimeError("Qwen2.5-VL 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_qwen_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"Qwen2.5-VL 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
# --- 새로 추가된 Idefics3 답변 생성 함수 ---
def _run_idefics3_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """Idefics3 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 대화형 프롬프트 및 이미지 준비
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"

    content = []
    pil_images = []
    
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)

    for img_path in image_paths:
        try:
            pil_images.append(Image.open(img_path).convert("RGB"))
            content.append({"type": "image"})
        except FileNotFoundError:
            logging.warning(f"Idefics3 모델을 위한 이미지 파일을 찾을 수 없습니다: {img_path}")

    content.append({"type": "text", "text": user_prompt_text})
    messages = [{"role": "user", "content": content}]

    # 2. 입력 데이터 처리 (apply_chat_template 사용)
    prompt = idefics_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = idefics_processor(text=prompt, images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 3. 텍스트 생성 (모델 추론)
    with torch.no_grad():
        generated_ids = idefics_model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    # 4. 결과 디코딩
    generated_texts = idefics_processor.batch_decode(generated_ids, skip_special_tokens=True)
    
    # 생성된 텍스트에서 프롬프트와 assistant 응답 부분만 추출
    # 예: "User: ... \nAssistant: {답변}"
    final_answer = generated_texts[0].split("Assistant:")[-1].strip()

    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids[0]) - prompt_tokens
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return final_answer, token_usage

@register_vlm("idefics3-8b")
async def generate_answer_idefics3(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 Idefics3 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not idefics_model or not idefics_processor:
        raise RuntimeError("Idefics3 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_idefics3_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"Idefics3 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# --- 새로 추가된 moondream3 답변 생성 함수 ---
def _run_moondream3_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """moondream3 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 프롬프트 구성
    system_like_prompt = (
        "Accurately answer questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    full_prompt = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        full_prompt += f"\n\n[Text Context]\n{context.markdown_content}"

    # 2. 이미지 로드
    pil_image = None
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)

    # moondream은 단일 이미지를 사용하므로 첫 번째 이미지만 처리
    if image_paths:
        try:
            pil_image = Image.open(image_paths[0]).convert("RGB")
        except FileNotFoundError:
            logging.warning(f"moondream3 모델을 위한 이미지 파일을 찾을 수 없습니다: {image_paths[0]}")

    # 3. 텍스트 생성 (모델 추론)
    # moondream은 이미지 없이 텍스트만으로도 질문 가능
    result = moondream_model.query(image=pil_image, question=full_prompt, reasoning=True)
    final_answer = result.get("answer", "")

    # 4. 토큰 사용량 추정
    # moondream은 토크나이저를 내장하고 있음
    prompt_tokens = len(moondream_model.tokenizer.encode(full_prompt))
    completion_tokens = len(moondream_model.tokenizer.encode(final_answer))
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

    return final_answer, token_usage

@register_vlm("moondream3")
async def generate_answer_moondream3(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 moondream3 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not moondream_model:
        raise RuntimeError("moondream3 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_moondream3_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"moondream3 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# --- 새로 추가된 DeepSeek-VL2 답변 생성 함수 ---
def _run_deepseek_vl2_sync(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """DeepSeek-VL2 모델 추론을 위한 동기 헬퍼 함수"""
    # 1. 대화형 프롬프트 및 이미지 경로 구성
    system_like_prompt = (
        "You are an AI assistant who accurately answers questions based on the provided context (text and images). "
        "Do not answer with information that is not in the context. "
        "Your final answer MUST be written in Korean."
    )
    user_prompt = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]:
        user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"

    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)

    # Deepseek-VL 형식에 맞게 content 구성
    content = "<image>" * len(image_paths) + "\n" + user_prompt
    
    conversation = [
        {"role": "<|User|>", "content": content, "images": image_paths},
        {"role": "<|Assistant|>", "content": ""}
    ]

    # 2. 입력 데이터 처리
    pil_images = load_pil_images(conversation)
    prepare_inputs = deepseek_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(deepseek_model.device)

    # 3. 텍스트 생성 (모델 추론)
    inputs_embeds = deepseek_model.prepare_inputs_embeds(**prepare_inputs)
    
    with torch.no_grad():
        outputs = deepseek_model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=deepseek_processor.tokenizer.eos_token_id,
            bos_token_id=deepseek_processor.tokenizer.bos_token_id,
            eos_token_id=deepseek_processor.tokenizer.eos_token_id,
            max_new_tokens=1024,
            do_sample=False,
            use_cache=True
        )

    # 4. 결과 디코딩
    final_answer = deepseek_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

    # 5. 토큰 사용량 추정
    prompt_tokens = len(prepare_inputs.input_ids[0])
    completion_tokens = len(outputs[0]) - prompt_tokens
    token_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": len(outputs[0]),
    }

    return final_answer, token_usage


@register_vlm("deepseek-vl2")
async def generate_answer_deepseek_vl2(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    """로컬 DeepSeek-VL2 모델을 사용하여 답변을 생성합니다. (비동기 래퍼)"""
    if not deepseek_model or not deepseek_processor:
        raise RuntimeError("DeepSeek-VL2 모델이 초기화되지 않았습니다.")
        
    try:
        answer, token_usage = await asyncio.to_thread(
            _run_deepseek_vl2_sync, question, context, model_combo
        )
        return answer, token_usage
    except Exception as e:
        logging.error(f"DeepSeek-VL2 추론 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


async def generate_answer(
    model_name: str, 
    question: Dict, 
    context: PageContext, 
    model_combo: str
) -> Tuple[str, Dict]:
    if model_name not in VLM_REGISTRY:
        raise ValueError(f"등록되지 않은 모델: {model_name}. 사용 가능: {list(VLM_REGISTRY.keys())}")
    
    logging.info(f"모델 '{model_name}'을 사용하여 '{model_combo}' 조합으로 답변 생성을 시작합니다.")
    answer_function = VLM_REGISTRY[model_name]
    
    return await answer_function(question, context, model_combo)

if __name__ == '__main__':
    logging.info(f"AnswerGen 모듈 로드 완료. 등록된 VLM 모델: {list(VLM_REGISTRY.keys())}")