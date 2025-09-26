import logging
from pathlib import Path
from pdf2image import convert_from_path
from typing import List, Dict, Any
import json
import dataclasses
import argparse
import asyncio
from collections import defaultdict
import itertools
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

async def process_single_answer_generation(
    semaphore: asyncio.Semaphore,
    job_info: Dict
) -> Dict:
    """하나의 질문-모델-조합에 대한 답변 '생성'만 처리하는 워커"""
    async with semaphore:
        try:
            answer, token_usage = await generate_answer(
                model_name=job_info['model'],
                question=job_info['question'],
                context=job_info['context'],
                model_combo=job_info['combo']
            )
            job_info.update({
                "status": "success",
                "answer": answer,
                "token_usage": token_usage
            })
            return job_info
        except Exception as e:
            logging.error(f"Answer generation failed for Q: '{job_info['question'].get('question', '')[:30]}...' on model {job_info['model']}/{job_info['combo']}: {e}", exc_info=False)
            job_info.update({"status": "failure", "error": str(e)})
            return job_info

# ==========================================================================
# [변경] 심사 단계를 위한 비동기 헬퍼 함수 추가
# ==========================================================================
async def process_single_scoring_judge(
    semaphore: asyncio.Semaphore,
    ans_data: Dict,
    judge_model: str
) -> Dict:
    """점수제 심사 단일 작업을 처리하고 결과를 원래 데이터와 함께 반환합니다."""
    async with semaphore:
        try:
            score = await judge_answer(
                judge_model_name=judge_model,
                question=ans_data['question'],
                context=ans_data['context'],
                answer=ans_data['answer']
            )
            return {"ans_data": ans_data, "judge_model": judge_model, "score": score}
        except Exception as e:
            logging.error(f"Scoring failed for judge {judge_model}: {e}", exc_info=False)
            return {"ans_data": ans_data, "judge_model": judge_model, "score": {"note": f"Error: {e}"}}

async def process_single_pairwise_judge(
    semaphore: asyncio.Semaphore,
    ans_a: Dict,
    ans_b: Dict,
    judge_model: str
) -> Dict:
    """쌍대 비교 심사 단일 작업을 처리하고 결과를 원래 데이터와 함께 반환합니다."""
    async with semaphore:
        try:
            result = await judge_answer(
                judge_model_name=judge_model,
                question=ans_a['question'],
                context=ans_a['context'],
                answer={'A': ans_a['answer'], 'B': ans_b['answer']}
            )
            return {"ans_a": ans_a, "ans_b": ans_b, "result": result}
        except Exception as e:
            logging.error(f"Pairwise judging failed for {ans_a['model']}_{ans_a['combo']} vs {ans_b['model']}_{ans_b['combo']}: {e}", exc_info=False)
            return {"ans_a": ans_a, "ans_b": ans_b, "result": {"winner": "error", "reason": f"Error: {e}"}}

# --------------------------------------------------------------------------
# 메인 비동기 실행 함수
# --------------------------------------------------------------------------
async def async_main(args):
    # ... (파일 처리 및 질문/답변 생성 부분은 기존과 동일) ...
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
            shutil.copy(file_path, pages_dir / f"page_01{file_path.suffix}")
        
        logging.info(f"'{pages_dir.name}' 폴더의 모든 페이지에 대한 배치 전처리를 시작합니다.")
        all_contexts = batch_process_pages(pages_dir, doc_output_dir, use_captioning=args.captioning)
        context_map = {ctx.page_id: ctx for ctx in all_contexts}
        
        if args.save_context:
            context_dir = doc_output_dir / "preprocessed_context"
            context_dir.mkdir(exist_ok=True)
            for context in all_contexts:
                with open(context_dir / f"{context.page_id}.json", 'w', encoding='utf-8') as f:
                    json.dump(dataclasses.asdict(context), f, ensure_ascii=False, indent=4)
        
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

                tasks_this_pass = [
                    asyncio.create_task(process_single_question_generation(semaphore, context_job))
                    for context_job in q_jobs_to_process
                ]
                results_this_pass = await asyncio.gather(*tasks_this_pass)

                successful_this_pass = [res for res in results_this_pass if res['status'] == 'success']
                failed_this_pass = [res for res in results_this_pass if res['status'] == 'failure']
                
                successful_q_results.extend(successful_this_pass)
                
                if failed_this_pass:
                    logging.warning(f"질문 생성 {attempt + 1}차 시도 후 {len(failed_this_pass)}개의 작업이 실패했습니다.")
                    q_jobs_to_process = [res['context'] for res in failed_this_pass]
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
            # --- 1. 모든 답변 일괄 생성 ---
            logging.info("--- 모든 답변 일괄 생성 시작 ---")
            answer_job_defs = []
            for model in args.models:
                for context in all_contexts:
                    q_path = doc_output_dir / "generated_questions" / f"{context.page_id}_questions.json"
                    if not q_path.exists(): continue
                    with open(q_path, 'r', encoding='utf-8') as f:
                        questions = json.loads(f.read().strip() or "[]")
                    for q in questions:
                        for combo in args.combos:
                            answer_job_defs.append({
                                "doc_id": file_path.stem, "page_id": context.page_id,
                                "context": context, "question": q,
                                "model": model, "combo": combo,
                            })
            
            semaphore = asyncio.Semaphore(args.concurrency)
            answer_tasks = []
            for job in answer_job_defs:
                task = asyncio.create_task(process_single_answer_generation(semaphore, job))
                answer_tasks.append(task)
                await asyncio.sleep(args.interval) # 답변 생성 시에도 interval 적용

            generated_answers_results = await asyncio.gather(*answer_tasks)
            
            all_generated_answers = [res for res in generated_answers_results if res['status'] == 'success']
            logging.info(f"총 {len(all_generated_answers)}개의 답변 생성을 완료했습니다.")

            # --- 2. 질문별로 답변 그룹화 ---
            answers_by_question = defaultdict(list)
            for ans_data in all_generated_answers:
                q_key = (ans_data['doc_id'], ans_data['page_id'], ans_data['question']['question'])
                answers_by_question[q_key].append(ans_data)

            eval_dir = doc_output_dir / "evaluation_results"
            eval_dir.mkdir(exist_ok=True)

            # ========================================================================
            # [변경] '점수제' 평가 실행 로직 수정
            # ========================================================================
            if 'scoring' in args.eval_modes:
                logging.info(f"--- 점수제(Scoring) 평가 시작 (심사관: {args.scoring_judges}, 동시성: {args.concurrency}, 간격: {args.interval}초) ---")
                
                # Semaphore를 사용하여 동시 실행 수를 제어
                semaphore = asyncio.Semaphore(args.concurrency)
                scoring_tasks = []
                
                # 처리할 모든 심사 작업을 정의
                scoring_jobs = [
                    (ans_data, judge_model)
                    for ans_data in all_generated_answers
                    for judge_model in args.scoring_judges
                ]

                for ans_data, judge_model in scoring_jobs:
                    # 각 작업을 헬퍼 함수를 통해 비동기 태스크로 생성
                    task = asyncio.create_task(
                        process_single_scoring_judge(semaphore, ans_data, judge_model)
                    )
                    scoring_tasks.append(task)
                    # 각 작업 생성 후 interval 만큼 대기하여 요청 속도 조절
                    await asyncio.sleep(args.interval)

                # 모든 점수제 평가 작업이 끝날 때까지 기다림
                judging_results = await asyncio.gather(*scoring_tasks)
                
                # 생성된 답변 데이터에 점수 결과(judgements)를 추가
                for result in judging_results:
                    ans_data = result['ans_data']
                    judge_model = result['judge_model']
                    score = result['score']
                    
                    if 'judgements' not in ans_data:
                        ans_data['judgements'] = {}
                    ans_data['judgements'][judge_model] = dataclasses.asdict(score) if dataclasses.is_dataclass(score) else score
                
                scoring_file_path = eval_dir / "scoring_results.jsonl"
                with open(scoring_file_path, 'w', encoding='utf-8') as f_out:
                    for result in all_generated_answers:
                        # context는 매우 크므로 최종 결과 파일에서는 제외
                        result_copy = result.copy()
                        del result_copy['context']
                        f_out.write(json.dumps(result_copy, ensure_ascii=False) + '\n')
                logging.info(f"점수제 평가 완료. 저장: {scoring_file_path}")

            # ========================================================================
            # [변경] '쌍대 비교' 평가 실행 로직 수정
            # ========================================================================
            if 'pairwise' in args.eval_modes:
                logging.info(f"--- 쌍대 비교(Pairwise) 평가 시작 (심사관: {args.pairwise_judge}, 동시성: {args.concurrency}, 간격: {args.interval}초) ---")

                # Semaphore를 사용하여 동시 실행 수를 제어
                semaphore = asyncio.Semaphore(args.concurrency)
                pairwise_tasks = []

                # 처리할 모든 쌍대 비교 작업을 정의
                pairwise_jobs = [
                    (ans_a, ans_b)
                    for q_key, answers_group in answers_by_question.items()
                    if len(answers_group) >= 2
                    for ans_a, ans_b in itertools.combinations(answers_group, 2)
                ]

                for ans_a, ans_b in pairwise_jobs:
                    # 각 작업을 헬퍼 함수를 통해 비동기 태스크로 생성
                    task = asyncio.create_task(
                        process_single_pairwise_judge(semaphore, ans_a, ans_b, args.pairwise_judge)
                    )
                    pairwise_tasks.append(task)
                    # 각 작업 생성 후 interval 만큼 대기하여 요청 속도 조절
                    await asyncio.sleep(args.interval)
                
                # 모든 쌍대 비교 작업이 끝날 때까지 기다림
                pairwise_results = await asyncio.gather(*pairwise_tasks)
                
                pairwise_results_formatted = []
                for res in pairwise_results:
                    ans_a, ans_b, result = res['ans_a'], res['ans_b'], res['result']
                    pairwise_results_formatted.append({
                        'question': ans_a['question']['question'],
                        'page_id': ans_a['page_id'],
                        'model_A': f"{ans_a['model']}_{ans_a['combo']}",
                        'model_B': f"{ans_b['model']}_{ans_b['combo']}",
                        'winner': result.get('winner', 'error'),
                        'reason': result.get('reason', 'N/A')
                    })

                pairwise_file_path = eval_dir / "pairwise_results.jsonl"
                with open(pairwise_file_path, 'w', encoding='utf-8') as f_out:
                    for result in pairwise_results_formatted:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                logging.info(f"쌍대 비교 평가 완료. 저장: {pairwise_file_path}")
        
        logging.info(f"--- 문서 처리 완료: {file_path.name} ---")


# --------------------------------------------------------------------------
# 스크립트 실행 지점
# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="문서 처리 및 평가 파이프라인을 실행합니다.")
    
    parser.add_argument("--input", type=str, default="data/", help="처리할 파일 또는 폴더의 경로.")
    parser.add_argument("--output", type=str, default="results/", help="결과물을 저장할 최상위 폴더 경로.")
    parser.add_argument("--concurrency", type=int, default=3, help="동시에 실행할 최대 작업 수. Rate Limit을 피하기 위해 3~5 정도로 낮게 설정하는 것을 권장합니다.")
    parser.add_argument("--interval", type=float, default=2.0, help="각 작업 요청 사이의 최소 대기 시간 (초). Rate Limit을 피하기 위해 1.5초 이상으로 설정하는 것을 권장합니다.")
    parser.add_argument("--save_context", action="store_true", help="전처리된 context를 JSON 파일로 저장합니다.")
    parser.add_argument("--captioning", action="store_true", help="이미지 캡셔닝을 활성화합니다.")
    parser.add_argument("--generate_questions", action="store_true", help="질문 생성을 실행합니다.")
    parser.add_argument("--generate_answers", action="store_true", help="답변 생성 및 심사를 실행합니다.")
    parser.add_argument("--models", nargs='+', default=["gpt-4o"], help="답변 생성에 사용할 모델 목록.")
    parser.add_argument("--combos", nargs='+', default=["M1", "M2", "M3"], help="사용할 컨텍스트 조합.")
    
    parser.add_argument("--eval_modes", nargs='+', choices=['scoring', 'pairwise'], default=['scoring'], help="실행할 평가 모드. 예: scoring pairwise")
    parser.add_argument("--scoring_judges", nargs='+', default=["gpt-4o_fact_checker", "gpt-4o_groundedness_analyst", "gpt-4o_clarity_specialist"], help="점수제 평가에 사용할 심사관 목록.")
    parser.add_argument("--pairwise_judge", type=str, default="gpt-4o_pairwise_judge", help="쌍대 비교 평가에 사용할 심사관.")
    
    parser.add_argument("--run_analysis", action="store_true", help="결과 분석을 실행합니다.")
    
    args = parser.parse_args()
    
    asyncio.run(async_main(args))

    if args.run_analysis:
        logging.info("최종 분석 단계를 시작합니다.")
        # analysis.run_analysis_stage(Path(args.output))
        pass

    logging.info("모든 파이프라인 작업이 완료되었습니다.")