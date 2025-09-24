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
load_dotenv()

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
    source_image_path: str = ""
    markdown_content: str = ""
    image_paths: List[str] = field(default_factory=list)
    image_captions: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

# --------------------------------------------------------------------------
# 2. 외부 프로세스 호출 및 모델 실행
# --------------------------------------------------------------------------
def run_mineru_on_directory(input_dir: Path, mineru_output_dir: Path):
    """
    지정된 입력 '디렉토리'에 포함된 모든 이미지에 대해 'mineru' CLI를 실행합니다.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"입력 디렉토리가 존재하지 않습니다: {input_dir}")

    mineru_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "mineru",
        "-p", str(input_dir),
        "-o", str(mineru_output_dir),
        "--backend", "pipeline",
        "--lang", "korean"
    ]
    logging.info(f"실행 명령어: {' '.join(command)}")
    
    try:
        # 긴 OCR 프로세스를 위해 timeout을 늘려줄 수 있습니다. (예: timeout=1800)
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
        logging.info(f"MinerU 배치 프로세스가 성공적으로 완료되었습니다. (입력: {input_dir.name})")
    except subprocess.CalledProcessError as e:
        logging.error(f"MinerU 실행 중 오류 발생:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        raise
    except Exception as e:
        raise RuntimeError(f"MinerU 실행 중 예외 발생: {e}")

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
# 3. 메인 전처리 함수 (배치 처리용)
# --------------------------------------------------------------------------
def batch_process_pages(pages_dir: Path, doc_output_dir: Path, use_captioning: bool = False) -> List[PageContext]:
    """
    페이지 이미지들이 담긴 디렉토리를 통째로 MinerU로 처리하고,
    결과를 파싱하여 각 페이지에 대한 PageContext 객체 리스트를 반환합니다.
    """
    if not pages_dir.exists() or not any(pages_dir.iterdir()):
        logging.warning(f"처리할 페이지 이미지가 '{pages_dir}'에 없습니다.")
        return []
        
    # 1. MinerU를 전체 디렉토리에 대해 단 한 번 실행
    mineru_base_output_dir = doc_output_dir / "mineru_raw_output"
    run_mineru_on_directory(pages_dir, mineru_base_output_dir)

    all_page_contexts = []
    
    # 2. 원본 페이지 이미지 목록을 순서대로 가져옴
    source_image_paths = sorted(list(pages_dir.glob("*.png")) + list(pages_dir.glob("*.jpg")))

    # 3. MinerU 결과물 디렉토리에서 각 페이지의 결과물을 찾아 파싱
    for source_image_path in source_image_paths:
        page_id = source_image_path.stem
        logging.info(f"페이지 '{page_id}'의 MinerU 결과물 파싱을 시작합니다.")

        # MinerU는 입력 디렉토리 구조를 그대로 따라감: {output}/{input_dir_name}/{page_id}/auto
        result_folder = mineru_base_output_dir / page_id / "auto"
        
        md_content = ""
        md_path = result_folder / f"{page_id}.md"
        logging.debug("="*80)
        logging.debug(f"DEBUG: 스크립트가 찾고 있는 Markdown 파일 경로:")
        logging.debug(md_path)
        logging.debug(f"DEBUG: 해당 경로에 파일이 존재하는지? -> {md_path.exists()}")
        logging.debug("="*80)
        if md_path.exists():
            md_content = md_path.read_text(encoding='utf-8')
        else:
            logging.warning(f"Markdown 파일을 찾을 수 없습니다: {md_path}")

        image_paths = []
        images_dir = result_folder / "images"
        if images_dir.exists():
            image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            image_paths = [str(f.resolve()) for f in image_files]

        context = PageContext(
            page_id=page_id,
            source_image_path=str(source_image_path.resolve()),
            markdown_content=md_content,
            image_paths=image_paths
        )
        logging.info(f"'{page_id}' 결과 파싱 완료. Markdown: {len(md_content)}자, 이미지: {len(image_paths)}개.")

        # 4. 캡셔닝
        if use_captioning and context.image_paths:
            logging.info(f"'{page_id}' 이미지 캡셔닝 시작...")
            for img_path in context.image_paths:
                caption = generate_caption_for_image(img_path)
                context.image_captions[img_path] = caption
        
        all_page_contexts.append(context)

    logging.info(f"총 {len(all_page_contexts)}개 페이지의 전처리를 모두 완료했습니다.")
    return all_page_contexts