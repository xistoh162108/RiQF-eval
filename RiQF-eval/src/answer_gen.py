# src/answer_gen.py

import logging
import os
import base64
from typing import List, Dict, Any, Callable, Tuple
import gc  # 가비지 컬렉션 모듈
import asyncio

# --- API 클라이언트 및 관련 라이브러리 임포트 ---
import google.generativeai as genai
from anthropic import AsyncAnthropic, RateLimitError as AnthropicRateLimitError
from openai import AsyncOpenAI, RateLimitError as OpenAIRateLimitError
from PIL import Image
import torch

# --- Hugging Face 로컬 모델 라이브러리 임포트 ---
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
    AutoModelForVision2Seq,
    AutoProcessor as IdeficsAutoProcessor,
    AutoModelForCausalLM
)
from qwen_vl_utils import process_vision_info

# --- 재시도(Retry) 라이브러리 임포트 ---
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# --- 다른 모듈 임포트 ---
from preprocess import PageContext
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------------------------------
# 0. 기본 설정 및 API 클라이언트 초기화
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

# ========================================================================
# 1. 로컬 모델 지연 로딩을 위한 관리자 클래스 정의
# ========================================================================
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else torch.bfloat16
logging.info(f"선택된 장치(Device): {DEVICE}, 데이터 타입(Dtype): {DTYPE}")

class LocalModelManager:
    """로컬 Hugging Face 모델을 지연 로딩하고 메모리를 관리하는 클래스"""
    def __init__(self):
        self._loaded_models = {}  # {'model_name': {'model': ..., 'processor': ...}}

    def clear(self):
        """로드된 모든 모델을 메모리에서 해제하고 캐시를 비웁니다."""
        if not self._loaded_models:
            return
        
        logging.info(f"기존에 로드된 모델 {list(self._loaded_models.keys())}을(를) 메모리에서 해제합니다...")
        for model_info in self._loaded_models.values():
            del model_info['model']
            if 'processor' in model_info: del model_info['processor']
            if 'tokenizer' in model_info: del model_info['tokenizer']
        
        self._loaded_models.clear()
        gc.collect()
        
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        elif DEVICE == 'mps':
            torch.mps.empty_cache()
        logging.info("메모리 해제 및 캐시 정리 완료.")

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """모델 이름으로 모델과 프로세서를 가져옵니다. 없으면 로드합니다."""
        if model_name in self._loaded_models:
            logging.info(f"캐시에서 '{model_name}' 모델을 재사용합니다.")
            return self._loaded_models[model_name]

        self.clear()

        logging.info(f"'{model_name}' 모델 로딩을 시작합니다...")
        try:
            model, processor, tokenizer = None, None, None

            if model_name == "llama-3.2-vision":
                model_id = "meta-llama/Llama-3.2-11B-Vision"
                model = MllamaForConditionalGeneration.from_pretrained(model_id, torch_dtype=DTYPE, device_map="auto").eval()
                processor = MllamaAutoProcessor.from_pretrained(model_id)
            
            elif model_name == "paligemma2":
                model_id = "google/paligemma2-3b-pt-896"
                model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, dtype=DTYPE, device_map="auto").eval()
                processor = PaliGemmaProcessor.from_pretrained(model_id)
                processor.image_processor.do_resize = True
                processor.image_processor.size = {"height": 896, "width": 896}

            elif model_name == "smolvlm":
                model_id = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
                model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPE, device_map="auto").eval()
                processor = SmolVLMAutoProcessor.from_pretrained(model_id)
            
            elif model_name == "internvl-8b":
                model_id = "OpenGVLab/InternVL2_5-8B"
                model = AutoModel.from_pretrained(model_id, torch_dtype=DTYPE, low_cpu_mem_usage=True, trust_remote_code=True).to(DEVICE).eval()
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)

            elif model_name == "qwen2.5-vl":
                model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto").eval()
                processor = QwenAutoProcessor.from_pretrained(model_id)

            elif model_name == "idefics3-8b":
                model_id = "HuggingFaceM4/Idefics3-8B-Llama3"
                model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=DTYPE, device_map="auto").eval()
                processor = IdeficsAutoProcessor.from_pretrained(model_id)
            
            else:
                raise ValueError(f"정의되지 않은 로컬 모델 이름: {model_name}")

            loaded_components = {'model': model}
            if processor: loaded_components['processor'] = processor
            if tokenizer: loaded_components['tokenizer'] = tokenizer

            self._loaded_models[model_name] = loaded_components
            logging.info(f"'{model_name}' 모델 로딩 완료.")
            return self._loaded_models[model_name]

        except Exception as e:
            logging.error(f"'{model_name}' 모델 로딩 중 심각한 오류 발생: {e}", exc_info=True)
            self.clear()
            raise RuntimeError(f"'{model_name}' 모델을 로드할 수 없습니다.") from e

model_manager = LocalModelManager()

# ========================================================================
# 2. 모델별 답변 생성 함수 구현
# ========================================================================

VLM_REGISTRY: Dict[str, Callable] = {}
def register_vlm(name: str):
    def decorator(func: Callable): VLM_REGISTRY[name] = func; return func
    return decorator

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
        model = genai.GenerativeModel('gemini-1.5-pro-latest', system_instruction="You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. **Your final answer MUST be written in Korean.**")
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
    system_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. **Your final answer MUST be written in Korean.**"
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
        response = await async_anthropic_client.messages.create(model="claude-3-5-sonnet-20240620", system=system_prompt, messages=messages, max_tokens=1024, temperature=0.1)
        answer = response.content[0].text
        token_usage = {"prompt_tokens": response.usage.input_tokens, "completion_tokens": response.usage.output_tokens, "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
        return answer, token_usage
    except AnthropicRateLimitError as e:
        logging.warning(f"Claude Rate Limit. 재시도합니다... ({e})")
        raise e
    except Exception as e:
        logging.error(f"Claude Sonnet 4 API 최종 호출 실패: {e}")
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# --- 로컬 모델 동기/비동기 함수들 ---

def _run_llama_sync(model, processor, question, context, model_combo):
    user_prompt = f"Question: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"
    bos = processor.tokenizer.bos_token or ""
    image_token = "<|image|>"
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    prompt = f"{bos}{image_token}\n{system_like_prompt}\n\n{user_prompt}"
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    image_to_process = None
    if image_paths:
        try: image_to_process = Image.open(image_paths[0]).convert("RGB")
        except FileNotFoundError: logging.warning(f"Llama 모델을 위한 이미지 파일을 찾을 수 없습니다: {image_paths[0]}")
    inputs = processor(images=image_to_process, text=prompt, return_tensors="pt").to(model.device)
    eos_id = processor.tokenizer.eos_token_id
    output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True, eos_token_id=eos_id, pad_token_id=eos_id)
    gen_only_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    generated_text = processor.batch_decode(gen_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(gen_only_ids[0])
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return generated_text, token_usage

@register_vlm("llama-3.2-vision")
async def generate_answer_llama3_2_vision(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("llama-3.2-vision")
        answer, token_usage = await asyncio.to_thread(_run_llama_sync, components['model'], components['processor'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"Llama-3.2-Vision 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _run_paligemma_sync(model, processor, question, context, model_combo):
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    user_prompt = f"Question: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"
    prompt = f"<image>\nanswer ko {system_like_prompt}\n\n{user_prompt}"
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    image_to_process = None
    if image_paths:
        try: image_to_process = Image.open(image_paths[0]).convert("RGB")
        except FileNotFoundError: logging.warning(f"PaliGemma 모델을 위한 이미지 파일을 찾을 수 없습니다: {image_paths[0]}")
    inputs = processor(images=image_to_process, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    output_str = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    prompt_part = f"answer ko {system_like_prompt}\n\n{user_prompt}"
    generated_text = output_str.replace(prompt_part, "").strip()
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids[0]) - prompt_tokens
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return generated_text, token_usage

@register_vlm("paligemma2")
async def generate_answer_paligemma(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("paligemma2")
        answer, token_usage = await asyncio.to_thread(_run_paligemma_sync, components['model'], components['processor'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"PaliGemma 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _run_smolvlm_sync(model, processor, question, context, model_combo):
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"
    content = [{"type": "text", "text": user_prompt_text}]
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in reversed(image_paths): content.insert(0, {"type": "image", "path": img_path})
    conversation = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad(): output_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    answer_part = generated_text.split("assistant\n")
    final_answer = answer_part[1].strip() if len(answer_part) > 1 else generated_text.strip()
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(output_ids[0]) - prompt_tokens
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return final_answer, token_usage

@register_vlm("smolvlm")
async def generate_answer_smolvlm(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("smolvlm")
        answer, token_usage = await asyncio.to_thread(_run_smolvlm_sync, components['model'], components['processor'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"SmolVLM 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _run_internvl_sync(model, tokenizer, question, context, model_combo):
    pil_images = []
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in image_paths:
        try: pil_images.append(Image.open(img_path).convert("RGB"))
        except FileNotFoundError: logging.warning(f"InternVL 모델용 이미지를 찾을 수 없습니다: {img_path}")
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    image_tokens = '\n'.join(['<image>' for _ in pil_images])
    user_prompt = f"{image_tokens}\n{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt += f"\n\n[Text Context]\n{context.markdown_content}"
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    with torch.no_grad(): response = model.chat(tokenizer=tokenizer, pil_imgs=pil_images, question=user_prompt, generation_config=generation_config)
    prompt_tokens = len(tokenizer.encode(user_prompt))
    completion_tokens = len(tokenizer.encode(response))
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return response, token_usage

@register_vlm("internvl-8b")
async def generate_answer_internvl(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("internvl-8b")
        answer, token_usage = await asyncio.to_thread(_run_internvl_sync, components['model'], components['tokenizer'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"InternVL 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _run_qwen_sync(model, processor, question, context, model_combo):
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"
    content = [{"type": "text", "text": user_prompt_text}]
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in reversed(image_paths): content.insert(0, {"type": "image", "image": f"file://{os.path.abspath(img_path)}"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad(): generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    final_answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids_trimmed[0])
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return final_answer, token_usage

@register_vlm("qwen2.5-vl")
async def generate_answer_qwen(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("qwen2.5-vl")
        answer, token_usage = await asyncio.to_thread(_run_qwen_sync, components['model'], components['processor'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"Qwen2.5-VL 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _run_idefics3_sync(model, processor, question, context, model_combo):
    system_like_prompt = "You are an AI assistant who accurately answers questions based on the provided context (text and images). Do not answer with information that is not in the context. Your final answer MUST be written in Korean."
    user_prompt_text = f"{system_like_prompt}\n\nQuestion: {question['question']}"
    if model_combo in ["M3", "M4", "M5", "M6", "M7"]: user_prompt_text += f"\n\n[Text Context]\n{context.markdown_content}"
    content, pil_images = [], []
    image_paths = []
    if model_combo in ["M1", "M2", "M3", "M4"]:
        if context.source_image_path: image_paths.append(context.source_image_path)
    elif model_combo == "M5":
        if context.image_paths: image_paths.extend(context.image_paths)
    for img_path in image_paths:
        try:
            pil_images.append(Image.open(img_path).convert("RGB"))
            content.append({"type": "image"})
        except FileNotFoundError: logging.warning(f"Idefics3 모델용 이미지를 찾을 수 없습니다: {img_path}")
    content.append({"type": "text", "text": user_prompt_text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=pil_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad(): generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    final_answer = generated_texts[0].split("Assistant:")[-1].strip()
    prompt_tokens = len(inputs["input_ids"][0])
    completion_tokens = len(generated_ids[0]) - prompt_tokens
    token_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens}
    return final_answer, token_usage

@register_vlm("idefics3-8b")
async def generate_answer_idefics3(question: Dict, context: PageContext, model_combo: str) -> Tuple[str, Dict]:
    try:
        components = model_manager.get_model("idefics3-8b")
        answer, token_usage = await asyncio.to_thread(_run_idefics3_sync, components['model'], components['processor'], question, context, model_combo)
        return answer, token_usage
    except Exception as e:
        logging.error(f"Idefics3 답변 생성 중 오류 발생: {e}", exc_info=True)
        return f"오류: {e}", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

# ========================================================================
# 3. 메인 답변 생성 라우터 함수
# ========================================================================
async def generate_answer(
    model_name: str, 
    question: Dict, 
    context: PageContext, 
    model_combo: str
) -> Tuple[str, Dict]:
    """요청된 모델을 찾아 답변 생성을 위임하는 메인 함수"""
    if model_name not in VLM_REGISTRY:
        raise ValueError(f"등록되지 않은 모델: {model_name}. 사용 가능: {list(VLM_REGISTRY.keys())}")
    
    logging.info(f"모델 '{model_name}'을 사용하여 '{model_combo}' 조합으로 답변 생성을 시작합니다.")
    answer_function = VLM_REGISTRY[model_name]
    
    return await answer_function(question, context, model_combo)

if __name__ == '__main__':
    logging.info(f"AnswerGen 모듈 로드 완료. 등록된 VLM 모델: {list(VLM_REGISTRY.keys())}")
    logging.info("로컬 모델 관리자가 준비되었습니다. generate_answer() 함수를 호출하여 사용하세요.")
    # 예시:
    # async def main_test():
    #     # dummy_context와 dummy_question이 정의되어 있다고 가정
    #     print(await generate_answer("llama-3.2-vision", dummy_question, dummy_context, "M3"))
    #     print(await generate_answer("paligemma2", dummy_question, dummy_context, "M3"))
    # asyncio.run(main_test())