import google.generativeai as genai
from PIL import Image
import os

genai.configure(api_key = "GOOGLE_API_KEY")

# 모델 로드
model = genai.GenerativeModel("gemini-1.5-flash")

# 이미지 열기 (PIL로)
img = Image.open("/Users/bagjimin/Desktop/Second_Brain/1. Projects/HyperX/Parser/VLM_Test/test.jpg")

# 텍스트 + 이미지 동시 입력
response = model.generate_content(
    [img, "Describe this image. Speaker's name"]
)

print(response.text)