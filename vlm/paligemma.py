from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch

model_id = "google/paligemma2-3b-pt-896"

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.bfloat16

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, dtype=dtype, device_map="auto"
).eval()

processor = PaliGemmaProcessor.from_pretrained(model_id)

# SIGLIP 이미지 프로세서는 height/width가 필요
processor.image_processor.do_resize = True
processor.image_processor.size = {"height": 896, "width": 896}

image_path = "/Users/bagjimin/Desktop/Second_Brain/1. Projects/HyperX/Parser/VLM_Test/test.jpg"
image = Image.open(image_path).convert("RGB")

# 이미지 1장 → <image> 1개
prompt = "<image>\nanswer en What is shown in this picture? in detail."

# images 먼저, text 다음(중복 삽입 이슈 방지)
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

out = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(out.strip())