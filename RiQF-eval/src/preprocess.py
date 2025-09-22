# src/preprocess.py

import json
import logging
import os
import subprocess
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()  # .env 파일의 환경 변수를 os.environ에 로드

# --------------------------------------------------------------------------
# 0. 로깅 및 OpenAI 클라이언트 설정
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except TypeError:
    logging.error("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    client = None

# --------------------------------------------------------------------------
# 1. 데이터 구조 정의
# --------------------------------------------------------------------------
@dataclass
class PageContext:
    """
    페이지의 모든 추출된 정보를 담는 데이터 컨테이너.
    """
    page_id: str
    source_image_path: str = "" # VLM 입력 시 원본 전체 이미지를 참조하기 위한 경로
    markdown_content: str = ""
    image_paths: List[str] = field(default_factory=list) # MinerU가 추출한 하위 이미지들
    image_captions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# --------------------------------------------------------------------------
# 2. 외부 프로세스 호출 및 모델 실행
# --------------------------------------------------------------------------
def run_mineru_and_get_artifacts(input_file: Path, mineru_output_dir: Path) -> Dict[str, Any]:
    """
    지정된 출력 폴더에 'mineru' CLI를 실행하고, 생성된 결과물들을 수집하여 반환합니다.
    """
    if not input_file.exists():
        raise FileNotFoundError(f"입력 파일이 존재하지 않습니다: {input_file}")

    mineru_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mineru",
        "-p", str(input_file),
        "-o", str(mineru_output_dir),
        "--backend", "pipeline",  # 백엔드 지정
        "--lang", "korean"      # 언어 지정
    ]
    """
    command = [
        "mineru",
        "-p", str(input_file),
        "-o", str(mineru_output_dir),
        "--backend", "vlm-vllm-engine" # VLM 백엔드 지정
    ]
    """
    logging.info(f"실행 명령어: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info("MinerU 프로세스가 성공적으로 완료되었습니다.")
        
        result_folder = mineru_output_dir / input_file.stem / "auto"
        artifacts = {}
        
        md_path = result_folder / f"{input_file.stem}.md"
        if md_path.exists():
            artifacts['markdown'] = md_path.read_text(encoding='utf-8')
            logging.info(f"Markdown 파일 로드 성공: {md_path}")
        else:
            logging.warning(f"결과 파일 '{md_path}'을 찾을 수 없습니다.")
            artifacts['markdown'] = ""

        images_dir = result_folder / "images"
        image_paths = []
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            image_paths = [str(f.resolve()) for f in image_files]
        artifacts['image_paths'] = image_paths
        
        return artifacts

    except Exception as e:
        raise RuntimeError(f"MinerU 실행 또는 결과 수집 중 오류: {e}")

def generate_caption_for_image(image_path: str, model: str = "gpt-4o") -> str:
    """OpenAI Vision 모델을 사용하여 주어진 이미지에 대한 캡션을 생성합니다."""
    if not client:
        logging.warning("OpenAI 클라이언트가 초기화되지 않아 캡션 생성을 건너뜁니다.")
        return "캡션 생성 실패: API 클라이언트 미설정"

    logging.info(f"OpenAI API 호출하여 이미지 캡션 생성: {Path(image_path).name}")
    
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "이 이미지는 문서에서 추출되었습니다. 이미지의 내용을 간결하게 설명해주세요."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        caption = response.choices[0].message.content.strip()
        logging.info(f"생성된 캡션: {caption[:50]}...")
        return caption
    except Exception as e:
        logging.error(f"이미지 캡션 생성 중 OpenAI API 오류 발생: {e}")
        return "캡션 생성 중 오류 발생"

# --------------------------------------------------------------------------
# 3. 메인 전처리 함수
# --------------------------------------------------------------------------
def process_page(file_path: Path, doc_output_dir: Path, use_captioning: bool = False) -> PageContext:
    """
    MinerU를 실행하여 모든 결과물을 수집하고 풍부한 PageContext 객체를 생성합니다.
    """
    page_id = file_path.stem
    logging.info(f"페이지 '{page_id}'에 대한 전처리 작업을 시작합니다.")
    
    mineru_output_target_dir = doc_output_dir / "mineru_raw_output" / page_id
    
    mineru_artifacts = run_mineru_and_get_artifacts(file_path, mineru_output_target_dir)

    context = PageContext(
        page_id=page_id,
        source_image_path=str(file_path.resolve()), # 원본 이미지의 절대 경로 저장
        markdown_content=mineru_artifacts.get('markdown', ""),
        image_paths=mineru_artifacts.get('image_paths', [])
    )
    logging.info(f"MinerU 결과물 수집 완료. Markdown 길이: {len(context.markdown_content)}자, 이미지: {len(context.image_paths)}개.")

    if use_captioning and context.image_paths:
        logging.info("이미지 캡셔닝을 시작합니다.")
        for img_path in context.image_paths:
            caption = generate_caption_for_image(img_path)
            context.image_captions[img_path] = caption
        logging.info(f"총 {len(context.image_captions)}개 이미지의 캡션 생성을 완료했습니다.")
    else:
        logging.info("이미지 캡셔닝 기능이 비활성화되었거나, 추출된 이미지가 없습니다.")

    logging.info(f"페이지 '{page_id}' 전처리 완료.")
    return context

# --------------------------------------------------------------------------
# 4. 스크립트 실행 테스트
# --------------------------------------------------------------------------
if __name__ == '__main__':
    logging.info("전처리 스크립트(preprocess.py)를 직접 실행하여 테스트합니다.")
    
    # 이 스크립트를 직접 테스트하려면 아래 경로에 실제 PDF 또는 이미지 파일을 위치시켜야 합니다.
    test_file_path = Path("data/test.pdf")
    # 테스트 결과를 저장할 임시 폴더
    test_output_dir = Path("results_preprocess_test")

    if not test_file_path.exists():
        test_file_path.parent.mkdir(parents=True, exist_ok=True)
        test_file_path.write_text("Dummy content for test file.")
        logging.warning(f"테스트 파일 '{test_file_path}'이 없어 더미 파일을 생성했습니다.")
        logging.warning("정확한 테스트를 위해 유효한 PDF 또는 이미지 파일로 교체하십시오.")
    
    try:
        # main.py의 process_document와 유사한 로직으로 단일 파일 테스트
        # 여기서는 PDF 분할을 직접 수행하지 않고, 단일 페이지 파일(이미지 또는 1페이지 PDF)을 가정합니다.
        page_context_result = process_page(test_file_path, test_output_dir, use_captioning=True)
        
        print("\n" + "="*50)
        print("[전처리 테스트 최종 결과 요약]")
        print(f"페이지 ID: {page_context_result.page_id}")
        print(f"원본 이미지 경로: {page_context_result.source_image_path}")
        print(f"추출된 Markdown 콘텐츠 길이: {len(page_context_result.markdown_content)}")
        print(f"추출된 이미지 수: {len(page_context_result.image_paths)}")
        print(f"생성된 캡션 수: {len(page_context_result.image_captions)}")
        print("="*50)

    except Exception as e:
        logging.error("전처리 테스트 중단.", exc_info=True)