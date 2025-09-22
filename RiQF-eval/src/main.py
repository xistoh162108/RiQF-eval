# src/main.py

import logging
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict, Any
import json
import dataclasses
import argparse
import asyncio
import pandas as pd

# 각 모듈에서 비동기 버전의 함수들을 임포트합니다.
from preprocess import process_page, PageContext
from query_gen import generate_questions_for_page
from answer_gen import generate_answer
from judge import judge_answer, run_heuristic_checks
# import analysis # 필요 시 주석 해제

# --------------------------------------------------------------------------
# 로깅 설정
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --------------------------------------------------------------------------
# PDF 변환 함수
# --------------------------------------------------------------------------
def convert_pdf_to_images(pdf_path: Path, output_folder: Path) -> List[Path]:
    """PDF 파일을 페이지별 이미지로 변환합니다."""
    logging.info(f"PDF 파일 변환 시작: {pdf_path}")
    output_folder.mkdir(parents=True, exist_ok=True)
    try:
        images = convert_from_path(pdf_path, dpi=200, fmt='png', thread_count=4)
    except Exception as e:
        logging.error(f"pdf2image 실행 오류. Poppler가 설치 및 PATH에 등록되었는지 확인하십시오.")
        raise e
    
    image_paths = [output_folder / f"page_{i+1:02d}.png" for i in range(len(images))]
    for path, image in zip(image_paths, images):
        image.save(path, 'PNG')
    
    logging.info(f"총 {len(image_paths)}개 페이지를 이미지로 변환 완료. 저장 위치: {output_folder}")
    return image_paths

# --------------------------------------------------------------------------
# 비동기 처리 워커 함수
# --------------------------------------------------------------------------
async def process_single_evaluation(
    semaphore: asyncio.Semaphore,
    doc_id: str,
    context: PageContext,
    question: Dict,
    model: str,
    combo: str,
    judges_to_use: List[str]
) -> Dict:
    """하나의 질문-모델-조합에 대한 답변 생성 및 심사를 비동기로 처리하는 단일 작업 단위"""
    async with semaphore:
        try:
            answer, token_usage = await generate_answer(
                model_name=model, question=question, context=context, model_combo=combo
            )
            result_record = {
                "doc_id": doc_id, "page_id": context.page_id,
                "model": model, "combo": combo, "question": question,
                "answer": answer, "token_usage": token_usage,
            }
            result_record["heuristic_checks"] = run_heuristic_checks(answer, context)
            judge_tasks = [
                judge_answer(judge_model_name=j_model, question=question, context=context, answer=answer)
                for j_model in judges_to_use
            ]
            judge_scores = await asyncio.gather(*judge_tasks)
            judgements = {judge: dataclasses.asdict(score) for judge, score in zip(judges_to_use, judge_scores)}
            result_record["judgements"] = judgements
            return result_record
        except Exception as e:
            logging.error(f"Evaluation failed for Q: '{question.get('question', '')[:30]}...' on model {model}/{combo}: {e}", exc_info=False)
            return {
                "error": str(e),
                "doc_id": doc_id, "page_id": context.page_id, "model": model, "combo": combo, "question": question
            }

# --------------------------------------------------------------------------
# 메인 비동기 실행 함수
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 메인 비동기 실행 함수 (수정된 버전)
# --------------------------------------------------------------------------
async def async_main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_semaphore = asyncio.Semaphore(args.concurrency)

    files_to_process: List[Path] = []
    supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    if input_path.is_dir():
        for ext in supported_extensions:
            files_to_process.extend(input_path.glob(f"**/*{ext}"))
    elif input_path.is_file() and input_path.suffix.lower() in supported_extensions:
        files_to_process = [input_path]

    if not files_to_process:
        logging.warning("처리할 파일이 없습니다.")
        return

    logging.info(f"총 {len(files_to_process)}개의 파일을 처리합니다.")

    for file_path in files_to_process:
        logging.info(f"--- 문서 처리 시작: {file_path.name} ---")
        doc_output_dir = output_dir / file_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)
        
        page_image_paths = []
        if file_path.suffix.lower() == '.pdf':
            page_image_paths = convert_pdf_to_images(file_path, doc_output_dir / "pages")
        else:
            page_image_paths = [file_path]
        
        all_contexts = [process_page(p, doc_output_dir, use_captioning=args.captioning) for p in page_image_paths]
        context_map = {ctx.page_id: ctx for ctx in all_contexts}
        
        if args.save_context:
            context_dir = doc_output_dir / "preprocessed_context"
            context_dir.mkdir(exist_ok=True)
            for context in all_contexts:
                with open(context_dir / f"{context.page_id}.json", 'w', encoding='utf-8') as f:
                    json.dump(dataclasses.asdict(context), f, ensure_ascii=False, indent=4)
        
        if args.generate_questions:
            logging.info(f"총 {len(all_contexts)}개 페이지에 대한 질문 생성을 동시에 시작합니다.")
            question_gen_tasks = [generate_questions_for_page(ctx) for ctx in all_contexts]
            pages_with_questions = await asyncio.gather(*question_gen_tasks, return_exceptions=True)
            
            questions_dir = doc_output_dir / "generated_questions"
            questions_dir.mkdir(exist_ok=True)
            for context, questions in zip(all_contexts, pages_with_questions):
                if isinstance(questions, Exception):
                    logging.error(f"질문 생성 실패 ({context.page_id}): {questions}")
                    continue
                if questions:
                    q_path = questions_dir / f"{context.page_id}_questions.json"
                    with open(q_path, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, ensure_ascii=False, indent=4)
            logging.info("모든 페이지의 질문 생성을 완료했습니다.")

        # --- 3/4단계: 답변 생성 및 심사 (다단계 재시도 로직 적용) ---
        if args.generate_answers:
            # 1. 실행할 모든 작업의 '정보'를 먼저 정의
            job_definitions = []
            for context in all_contexts:
                q_path = doc_output_dir / "generated_questions" / f"{context.page_id}_questions.json"
                if not q_path.exists(): continue
                with open(q_path, 'r', encoding='utf-8') as f:
                    try:
                        questions = json.load(f)
                    except json.JSONDecodeError:
                        questions = []
                if not questions: continue
                
                for q in questions:
                    for model in args.models:
                        for combo in args.combos:
                            job_definitions.append({
                                "doc_id": file_path.stem, "context": context, "question": q,
                                "model": model, "combo": combo, "judges_to_use": args.judges
                            })
            
            all_successful_jobs = []
            jobs_to_process = job_definitions
            max_attempts = 5 # 최대 5차 시도

            for attempt in range(max_attempts):
                if not jobs_to_process:
                    logging.info(f"모든 작업이 성공하여 {attempt + 1}차 시도 전에 종료합니다.")
                    break

                # 시도 횟수에 따라 점진적으로 속도 감속
                is_first_attempt = (attempt == 0)
                concurrency = args.concurrency if is_first_attempt else max(1, args.concurrency - attempt*2)
                interval = args.interval if is_first_attempt else args.interval * (attempt + 1)
                semaphore = asyncio.Semaphore(concurrency)
                
                if is_first_attempt:
                    logging.info(f"--- 1차 실행 시작 ---")
                    logging.info(f"총 {len(jobs_to_process)}개 작업, 동시성: {concurrency}, 간격: {interval}초")
                else:
                    sleep_duration = 30 * attempt # 2차: 30초, 3차: 60초... 대기
                    logging.warning(f"--- {attempt + 1}차 재시도 시작 ({len(jobs_to_process)}개 남음) ---")
                    logging.info(f"동시성: {concurrency}, 간격: {interval}초. Rate Limit 초기화를 위해 {sleep_duration}초 대기합니다.")
                    await asyncio.sleep(sleep_duration)

                # 현재 시도할 작업(task) 목록 생성
                tasks_this_pass = []
                for job_info in jobs_to_process:
                    task = process_single_evaluation(semaphore, **job_info)
                    tasks_this_pass.append(task)
                    await asyncio.sleep(interval)
                
                # 현재 시도 실행
                results_this_pass = await asyncio.gather(*tasks_this_pass)

                # 성공/실패 분류
                successful_this_pass = [res for res in results_this_pass if "error" not in res]
                failed_this_pass = [res for res in results_this_pass if "error" in res]
                
                all_successful_jobs.extend(successful_this_pass)

                if failed_this_pass:
                    logging.warning(f"{attempt + 1}차 시도 후 {len(failed_this_pass)}개의 작업이 실패했습니다.")
                    # 다음 시도를 위해 실패한 작업들의 '정보'를 다시 구성
                    jobs_to_process = []
                    for failed_job in failed_this_pass:
                        context = context_map.get(failed_job['page_id'])
                        if context:
                            jobs_to_process.append({
                                "doc_id": failed_job['doc_id'], "context": context, "question": failed_job['question'],
                                "model": failed_job['model'], "combo": failed_job['combo'], "judges_to_use": args.judges
                            })
                else:
                    jobs_to_process = [] # 모든 작업 성공

            if jobs_to_process: # 최대 시도 후에도 실패한 작업
                logging.error(f"최대 {max_attempts}차 시도 후에도 {len(jobs_to_process)}개의 작업이 최종적으로 실패했습니다.")

            # --- 최종 결과 저장 ---
            logging.info(f"총 {len(all_successful_jobs)}개의 성공적인 결과를 파일에 저장합니다.")
            eval_dir = doc_output_dir / "evaluation_results"
            eval_dir.mkdir(exist_ok=True)
            eval_file_path = eval_dir / "eval_results.jsonl"
            with open(eval_file_path, 'w', encoding='utf-8') as f_out:
                for result in all_successful_jobs:
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            logging.info(f"모든 답변 생성 및 심사 완료. 저장: {eval_file_path}")
        
        logging.info(f"--- 문서 처리 완료: {file_path.name} ---")

# --------------------------------------------------------------------------
# 스크립트 실행 지점
# --------------------------------------------------------------------------
if __name__ == '__main__':
    # (argparse 설정은 이전과 동일)
    parser = argparse.ArgumentParser(description="문서 처리 및 평가 파이프라인을 실행합니다.")
    parser.add_argument("--input", type=str, default="data/", help="처리할 파일 또는 폴더의 경로.")
    parser.add_argument("--output", type=str, default="results/", help="결과물을 저장할 최상위 폴더 경로.")
    parser.add_argument("--concurrency", type=int, default=5, help="동시에 실행할 최대 작업 수.")
    parser.add_argument("--interval", type=float, default=1.5, help="각 작업 요청 사이의 최소 대기 시간 (초).")
    parser.add_argument("--save_context", action="store_true")
    parser.add_argument("--captioning", action="store_true")
    parser.add_argument("--generate_questions", action="store_true")
    parser.add_argument("--generate_answers", action="store_true")
    parser.add_argument("--judge_answers", action="store_true")
    parser.add_argument("--models", nargs='+', default=["gpt-4o"])
    parser.add_argument("--combos", nargs='+', default=["M1", "M2", "M3", "M4", "M5", "M6", "M7"])
    parser.add_argument("--judges", nargs='+', default=["gpt-4o", "gemini-1.5-pro", "claude-sonnet-4"])
    parser.add_argument("--run_analysis", action="store_true")
    
    args = parser.parse_args()
    asyncio.run(async_main(args))

    if args.run_analysis:
        logging.info("최종 분석 단계를 시작합니다.")
        # analysis.run_analysis_stage(Path(args.output))
        pass

    logging.info("모든 파이프라인 작업이 완료되었습니다.")