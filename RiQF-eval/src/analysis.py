# src/analysis.py

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter # Counter 임포트

import pandas as pd
import numpy as np
from scipy import stats
# import simpledorff # 더 이상 사용하지 않음

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def load_and_flatten_results(results_path: Path) -> pd.DataFrame:
    """
    eval_results.jsonl 파일을 로드하고, 분석하기 쉽도록 flatten된 'long-form' DataFrame으로 변환합니다.
    """
    if not results_path.exists():
        logging.error(f"결과 파일을 찾을 수 없습니다: {results_path}")
        return pd.DataFrame()

    records = []
    with open(results_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # judgements가 None이거나 비어있을 경우를 대비
            judgements = data.get('judgements')
            if not judgements:
                continue

            for judge_name, judgement in judgements.items():
                # LLM이 계산한 final_score 대신, 개별 점수를 사용해 우리가 직접 계산합니다.
                accuracy = judgement.get('accuracy')
                groundedness = judgement.get('groundedness')
                completeness = judgement.get('completeness')
                clarity = judgement.get('clarity')
                
                # 점수 중 하나라도 누락되면 건너뜁니다.
                if None in (accuracy, groundedness, completeness, clarity):
                    logging.warning(f"심사 점수 중 일부가 누락되어 이 레코드를 건너뜁니다: {data['doc_id']} - {data['page_id']}")
                    continue

                # 우리가 정의한 가중치를 사용하여 final_score를 재계산합니다.
                final_score = (
                    accuracy * 0.4 +
                    groundedness * 0.35 +
                    completeness * 0.15 +
                    clarity * 0.10
                )

                record = {
                    'doc_id': data['doc_id'],
                    'page_id': data['page_id'],
                    'model': data['model'],
                    'combo': data['combo'],
                    'question_text': data['question']['question'],
                    'question_type': data['question']['type'],
                    'question_difficulty': data['question']['difficulty'],
                    'answer': data['answer'],
                    'prompt_tokens': data.get('token_usage', {}).get('prompt_tokens'),
                    'completion_tokens': data.get('token_usage', {}).get('completion_tokens'),
                    'judge': judge_name,
                    'accuracy': accuracy,
                    'groundedness': groundedness,
                    'completeness': completeness,
                    'clarity': clarity,
                    'final_score': final_score # LLM이 계산한 값이 아닌, 우리가 계산한 값으로 덮어씁니다.
                }
                records.append(record)
    
    if not records:
        logging.warning("결과 파일에서 유효한 심사 데이터를 찾을 수 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df['answer_id'] = df.groupby(['doc_id', 'page_id', 'model', 'combo', 'question_text']).ngroup()
    return df

def calculate_ensemble_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 답변에 대한 심사관들의 점수를 정규화하고 앙상블 점수를 계산합니다.
    """
    if df.empty or 'final_score' not in df.columns:
        return df

    def robust_z_score(series):
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        if iqr == 0:
            return pd.Series(0, index=series.index)
        return (series - median) / iqr

    df['normalized_score'] = df.groupby('answer_id')['final_score'].transform(robust_z_score)
    ensemble_scores = df.groupby('answer_id')['normalized_score'].mean().rename('ensemble_score')
    df = df.merge(ensemble_scores, on='answer_id', how='left')
    
    logging.info("앙상블 점수 계산이 완료되었습니다.")
    return df

# ========================================================================
# Krippendorff's Alpha 계산 로직 직접 구현
# ========================================================================
def calculate_krippendorff_alpha(df: pd.DataFrame) -> float:
    """
    simpledorff 라이브러리 로직을 내장하여 심사관 간 일치도 (Krippendorff's Alpha)를 직접 계산합니다.
    """
    if df.empty or df['judge'].nunique() < 2:
        logging.warning("심사관이 2명 미만이어서 Krippendorff's Alpha를 계산할 수 없습니다.")
        return np.nan

    # --- simpledorff의 핵심 로직 시작 ---

    experiment_col = 'answer_id'
    annotator_col = 'judge'
    class_col = 'final_score'
    
    # Helper functions (라이브러리 코드에서 가져옴)
    def df_to_experiment_annotator_table(df, experiment_col, annotator_col, class_col):
        return df.pivot_table(index=annotator_col, columns=experiment_col, values=class_col, aggfunc="first")

    def make_value_by_unit_table_dict(experiment_annotator_df):
        data_by_exp = experiment_annotator_df.T.sort_index(axis=1).sort_index()
        table_dict = {}
        for exp, row in data_by_exp.iterrows():
            vals = row.dropna().values
            table_dict[exp] = Counter()
            for val in vals:
                table_dict[exp][val] += 1
        return table_dict

    def calculate_frequency_dicts(vbu_table_dict):
        vbu_df = pd.DataFrame.from_dict(vbu_table_dict, orient="index").T.sort_index(axis=0).sort_index(axis=1).fillna(0)
        ubv_df = vbu_df.T
        vbu_df_masked = ubv_df.mask(ubv_df.sum(1) == 1, other=0).T
        return dict(
            unit_freqs=vbu_df_masked.sum().to_dict(),
            class_freqs=vbu_df_masked.sum(1).to_dict(),
            total=vbu_df_masked.sum().sum(),
        )

    def interval_metric(x, y):
        return (x - y) ** 2

    def calculate_de(frequency_dicts, metric_fn):
        De = 0
        class_freqs = frequency_dicts["class_freqs"]
        class_names = list(class_freqs.keys())
        for i, c in enumerate(class_names):
            for k in class_names:
                De += class_freqs[c] * class_freqs[k] * metric_fn(c, k)
        return De

    def calculate_do(vbu_table_dict, frequency_dicts, metric_fn):
        Do = 0
        unit_freqs = frequency_dicts["unit_freqs"]
        unit_ids = list(unit_freqs.keys())
        for unit_id in unit_ids:
            unit_classes = list(vbu_table_dict[unit_id].keys())
            if unit_freqs[unit_id] < 2:
                pass
            else:
                weight = 1 / (unit_freqs[unit_id] - 1)
                for i, c in enumerate(unit_classes):
                    for k in unit_classes:
                        Do += vbu_table_dict[unit_id][c] * vbu_table_dict[unit_id][k] * weight * metric_fn(c, k)
        return Do

    # Main logic
    try:
        ea_table_df = df_to_experiment_annotator_table(df, experiment_col, annotator_col, class_col)
        vbu_table_dict = make_value_by_unit_table_dict(ea_table_df)
        frequency_dict = calculate_frequency_dicts(vbu_table_dict)
        
        # 점수는 연속형 변수이므로 interval_metric 사용
        observed_disagreement = calculate_do(vbu_table_dict, frequency_dict, interval_metric)
        expected_disagreement = calculate_de(frequency_dict, interval_metric)
        
        N = frequency_dict['total']
        if expected_disagreement == 0:
            return 1.0 # 완전 일치, 우연에 의한 불일치 없음
        
        alpha = 1 - (observed_disagreement / expected_disagreement) * (N - 1)
        
        logging.info(f"Krippendorff's Alpha: {alpha:.4f}")
        return alpha
    except Exception as e:
        logging.error(f"Krippendorff's Alpha 계산 중 오류 발생: {e}")
        return np.nan

def run_statistical_test(df: pd.DataFrame, combo1: str, combo2: str) -> Tuple[float, float]:
    df_filtered = df.dropna(subset=['ensemble_score'])
    scores1 = df_filtered[df_filtered['combo'] == combo1].drop_duplicates(subset='answer_id').set_index('answer_id')['ensemble_score']
    scores2 = df_filtered[df_filtered['combo'] == combo2].drop_duplicates(subset='answer_id').set_index('answer_id')['ensemble_score']
    common_answers = scores1.index.intersection(scores2.index)
    if len(common_answers) < 10:
        logging.warning(f"공통 답변 수가 너무 적어 ({len(common_answers)}개) 통계 검정을 건너뜁니다.")
        return np.nan, np.nan
    scores1 = scores1.loc[common_answers]
    scores2 = scores2.loc[common_answers]
    statistic, p_value = stats.wilcoxon(scores1, scores2)
    logging.info(f"'{combo1}' vs '{combo2}' Wilcoxon signed-rank test: p-value = {p_value:.4f}")
    return statistic, p_value