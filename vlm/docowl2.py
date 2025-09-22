from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch, os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = "mPLUG/DocOwl2"

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)

# 핵심 1: Config도 trust_remote_code=True
config = AutoConfig.from_pretrained(ckpt, trust_remote_code=True)

# 핵심 2: Model도 trust_remote_code=True
model = AutoModel.from_pretrained(
    ckpt,
    trust_remote_code=True,
    dtype=torch.float16,  # (경고 메시지 해결: torch_dtype 대신 dtype 사용)
    low_cpu_mem_usage=True,
)
model.to(device)

# DocOwl2 전처리 초기화
model.init_processor(tokenizer=tokenizer, basic_image_size=504, crop_anchors="grid_12")

def infer(images, query):
    msg = [{"role": "USER", "content": "<|image|>" * len(images) + query}]
    return model.chat(messages=msg, images=images, tokenizer=tokenizer)

images = ["./examples/docowl2_page0.png"]
print(infer(images, "한 줄 요약해줘"))