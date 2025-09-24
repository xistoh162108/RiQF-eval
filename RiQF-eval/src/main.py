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
import shutil

# 각 모듈에서 비동기 버전의 함수들을 임포트합니다.
from preprocess import batch_process_pages, PageContext 
from query_gen import generate_questions_for_page
from answer_gen import generate_answer
from judge import judge_answer, run_heuristic_checks

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
# 비동기 처리 워커 함수들
# --------------------------------------------------------------------------

# [신규] 질문 생성을 위한 비동기 워커 함수
async def process_single_question_generation(
    semaphore: asyncio.Semaphore,
    context: PageContext
) -> Dict:
    """단일 페이지에 대한 질문 생성을 안정적으로 처리합니다."""
    async with semaphore:
        try:
            questions = await generate_questions_for_page(context)
            return {"status": "success", "context": context, "questions": questions}
        except Exception as e:
            logging.error(f"Question generation failed for page {context.page_id}: {e}", exc_info=False)
            return {"status": "failure", "context": context, "error": str(e)}

# 답변 생성 및 심사를 위한 비동기 워커 함수 (기존과 동일)
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
async def async_main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        
        pages_dir = doc_output_dir / "pages"
        pages_dir.mkdir(exist_ok=True, parents=True)

        if file_path.suffix.lower() == '.pdf':
            convert_pdf_to_images(file_path, pages_dir)
        else:
            shutil.copy(file_path, pages_dir / file_path.name)
        
        logging.info(f"'{pages_dir.name}' 폴더의 모든 페이지에 대한 배치 전처리를 시작합니다.")
        all_contexts = batch_process_pages(pages_dir, doc_output_dir, use_captioning=args.captioning)
        context_map = {ctx.page_id: ctx for ctx in all_contexts}
        
        if args.save_context:
            context_dir = doc_output_dir / "preprocessed_context"
            context_dir.mkdir(exist_ok=True)
            for context in all_contexts:
                with open(context_dir / f"{context.page_id}.json", 'w', encoding='utf-8') as f:
                    json.dump(dataclasses.asdict(context), f, ensure_ascii=False, indent=4)
        
        # [변경] 질문 생성 단계에 재시도 및 동시성/간격 제어 로직 적용
        if args.generate_questions:
            q_jobs_to_process = all_contexts
            successful_q_results = []
            max_attempts = 5

            for attempt in range(max_attempts):
                if not q_jobs_to_process:
                    logging.info(f"모든 질문 생성이 성공하여 {attempt}차 시도 후 종료합니다.")
                    break
                
                is_first_attempt = (attempt == 0)
                concurrency = args.concurrency if is_first_attempt else max(1, args.concurrency - attempt)
                interval = args.interval if is_first_attempt else args.interval * (attempt + 1)
                semaphore = asyncio.Semaphore(concurrency)

                log_prefix = f"--- 질문 생성 {attempt + 1}차 실행 시작"
                if not is_first_attempt:
                    log_prefix += f" ({len(q_jobs_to_process)}개 재시도)"
                logging.info(log_prefix + f" (총 {len(q_jobs_to_process)}개, 동시성: {concurrency}, 간격: {interval}초) ---")
                
                if not is_first_attempt:
                    sleep_duration = 15 * attempt
                    logging.info(f"Rate Limit 초기화를 위해 {sleep_duration}초 대기합니다.")
                    await asyncio.sleep(sleep_duration)

                tasks_this_pass = []
                for context_job in q_jobs_to_process:
                    task = asyncio.create_task(process_single_question_generation(semaphore, context_job))
                    tasks_this_pass.append(task)
                    await asyncio.sleep(interval)
                
                results_this_pass = await asyncio.gather(*tasks_this_pass)

                successful_this_pass = [res for res in results_this_pass if res['status'] == 'success']
                failed_this_pass = [res for res in results_this_pass if res['status'] == 'failure']
                
                successful_q_results.extend(successful_this_pass)
                
                if failed_this_pass:
                    logging.warning(f"질문 생성 {attempt + 1}차 시도 후 {len(failed_this_pass)}개 작업이 실패했습니다.")
                    q_jobs_to_process = [failed['context'] for failed in failed_this_pass]
                else:
                    q_jobs_to_process = []
            
            if q_jobs_to_process:
                logging.error(f"최대 {max_attempts}차 시도 후에도 {len(q_jobs_to_process)}개의 질문 생성 작업이 최종 실패했습니다.")

            questions_dir = doc_output_dir / "generated_questions"
            questions_dir.mkdir(exist_ok=True)
            for result in successful_q_results:
                context = result['context']
                questions = result['questions']
                if questions:
                    q_path = questions_dir / f"{context.page_id}_questions.json"
                    with open(q_path, 'w', encoding='utf-8') as f:
                        json.dump(questions, f, ensure_ascii=False, indent=4)
            logging.info("모든 페이지의 질문 생성을 완료했습니다.")

        if args.generate_answers:
            job_definitions = []
            for model in args.models:
                for context in all_contexts:
                    q_path = doc_output_dir / "generated_questions" / f"{context.page_id}_questions.json"
                    if not q_path.exists(): continue
                    
                    with open(q_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        questions = json.loads(content) if content.strip() else []

                    if not questions: continue
                    
                    for q in questions:
                        for combo in args.combos:
                            job_definitions.append({
                                "doc_id": file_path.stem, "context": context, "question": q,
                                "model": model, "combo": combo, "judges_to_use": args.judges
                            })
            
            all_successful_jobs = []
            jobs_to_process = job_definitions
            max_attempts = 5

            for attempt in range(max_attempts):
                if not jobs_to_process:
                    logging.info(f"모든 작업이 성공하여 {attempt}차 시도 후 종료합니다.")
                    break

                is_first_attempt = (attempt == 0)
                concurrency = args.concurrency if is_first_attempt else max(1, args.concurrency - attempt*2)
                interval = args.interval if is_first_attempt else args.interval * (attempt + 1)
                semaphore = asyncio.Semaphore(concurrency)
                
                log_prefix = f"--- 답변/심사 {attempt + 1}차 실행 시작"
                if not is_first_attempt:
                    log_prefix += f" ({len(jobs_to_process)}개 재시도)"
                logging.info(log_prefix + f" (총 {len(jobs_to_process)}개, 동시성: {concurrency}, 간격: {interval}초) ---")

                if not is_first_attempt:
                    sleep_duration = 30 * attempt
                    logging.info(f"Rate Limit 초기화를 위해 {sleep_duration}초 대기합니다.")
                    await asyncio.sleep(sleep_duration)

                tasks_this_pass = []
                for job_info in jobs_to_process:
                    task = asyncio.create_task(process_single_evaluation(semaphore, **job_info))
                    tasks_this_pass.append(task)
                    await asyncio.sleep(interval)
                
                results_this_pass = await asyncio.gather(*tasks_this_pass)

                successful_this_pass = [res for res in results_this_pass if "error" not in res]
                failed_this_pass = [res for res in results_this_pass if "error" in res]
                
                all_successful_jobs.extend(successful_this_pass)

                if failed_this_pass:
                    logging.warning(f"답변/심사 {attempt + 1}차 시도 후 {len(failed_this_pass)}개의 작업이 실패했습니다.")
                    jobs_to_process = [
                        {**failed_job, 'context': context_map[failed_job['page_id']], 'judges_to_use': args.judges}
                        for failed_job in failed_this_pass
                    ]
                else:
                    jobs_to_process = []

            if jobs_to_process:
                logging.error(f"최대 {max_attempts}차 시도 후에도 {len(jobs_to_process)}개의 답변/심사 작업이 최종적으로 실패했습니다.")

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
    parser = argparse.ArgumentParser(description="문서 처리 및 평가 파이프라인을 실행합니다.")
    parser.add_argument("--input", type=str, default="data/", help="처리할 파일 또는 폴더의 경로.")
    parser.add_argument("--output", type=str, default="results/", help="결과물을 저장할 최상위 폴더 경로.")
    parser.add_argument("--concurrency", type=int, default=5, help="동시에 실행할 최대 작업 수.")
    parser.add_argument("--interval", type=float, default=1.5, help="각 작업 요청 사이의 최소 대기 시간 (초).")
    parser.add_argument("--save_context", action="store_true", help="전처리된 context를 JSON 파일로 저장합니다.")
    parser.add_argument("--captioning", action="store_true", help="이미지 캡셔닝을 활성화합니다.")
    parser.add_argument("--generate_questions", action="store_true", help="질문 생성을 실행합니다.")
    parser.add_argument("--generate_answers", action="store_true", help="답변 생성 및 심사를 실행합니다.")
    parser.add_argument("--models", nargs='+', default=["gpt-4o"], help="답변 생성에 사용할 모델 목록.")
    parser.add_argument("--combos", nargs='+', default=["M1", "M2", "M3", "M4", "M5", "M6", "M7"], help="사용할 컨텍스트 조합.")
    parser.add_argument("--judges", nargs='+', default=["gpt-4o"], help="답변 심사에 사용할 모델 목록.")
    parser.add_argument("--run_analysis", action="store_true", help="결과 분석을 실행합니다.")
    
    args = parser.parse_args()
    
    asyncio.run(async_main(args))

    if args.run_analysis:
        logging.info("최종 분석 단계를 시작합니다.")
        # analysis.run_analysis_stage(Path(args.output))
        pass

    logging.info("모든 파이프라인 작업이 완료되었습니다.")