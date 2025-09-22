from openai import OpenAI
from base64 import b64encode

client = OpenAI(api_key="OPENAI_API_KEY")

def encode_image(path):
    with open(path, "rb") as image_file:
        return b64encode(image_file.read()).decode('utf-8')
    
image_b64 = encode_image("/Users/bagjimin/Desktop/Second_Brain/1. Projects/HyperX/Parser/VLM_Test/test.jpg")

messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url",
         "image_url": {
            "url": f"data:image/jpeg;base64,{image_b64}",
            "detail": "auto"  # low / high / auto
         }}
    ]
}]

response = client.chat.completions.create(
    model="gpt-5-nano-2025-08-07",
    messages=messages,
    temperature=1.0
)
print(response.choices[0].message.content)