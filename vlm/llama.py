import requests
from PIL import Image
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
# import matplotlib.pyplot as plt

# 모델과 프로세서 로드
model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    dtype=torch.bfloat16,  # torch_dtype 대신 dtype 사용
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# 이미지 URL (자신의 URL로 교체 가능)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
path = "/Users/bagjimin/Desktop/Second_Brain/1. Projects/HyperX/Parser/VLM_Test/test.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(path).convert("RGB")  # 로컬 파일은 PIL로 열고, 반드시 RGB로 변환
# 이미지 출력 (matplotlib 사용)
"""
plt.imshow(image)
plt.axis('off')  # 축을 숨깁니다
plt.show()  # 이미지를 화면에 띄웁니다
"""

# 프롬프트 정의 (base 모델: 수동 포맷)
user_prompt = "Describe this image in detail."
bos = processor.tokenizer.bos_token or ""
image_token = "<|image|>"  # mllama 비전 전용 특별 토큰
prompt = f"{bos}{image_token}\n{user_prompt}"

# 입력 데이터 처리
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# 모델을 이용해 텍스트 생성 및 프롬프트 제외 디코딩
eos_id = processor.tokenizer.eos_token_id
output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,  # 결정적 출력
    use_cache=True,
    eos_token_id=eos_id,
    pad_token_id=eos_id
)
# Slice off the prompt part so we only decode new tokens
gen_only_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
generated_text = processor.batch_decode(
    gen_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)[0]
print(generated_text.strip())