# RiQF-eval: A Rigorous Quantitative Framework for Evaluating Multimodal Document QA Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![KAIST](https://img.shields.io/badge/KAIST-School%20of%20Computing-blue)](https://cs.kaist.ac.kr/)
[![HyperX](https://img.shields.io/badge/HyperX-AI-purple)](https://ax.hyper-x.ai/)

**Read this in other languages:** [**한국어 (Korean)**](./README-ko.md)

---

**RiQF-eval** is a comprehensive system for document parsing, question generation, and the rigorous evaluation of Vision-Language Models (VLMs). This repository presents a complete, reproducible methodological framework for the quantitative evaluation of multimodal document Question-Answering (QA) systems.

This project is a collaborative effort between **HyperX Inc.** and the **KAIST Lean Startup Program**.

## Abstract

This project introduces a comprehensive and reproducible framework for the quantitative evaluation of multimodal document QA systems. The protocol is designed to rigorously assess system performance across various input modality combinations by employing a fully automated pipeline. This pipeline encompasses: (1) standardized corpus preprocessing and test case generation; (2) controlled response generation under equitable conditions; and (3) a novel, multi-stage evaluation protocol featuring a robust ensemble of multiple Large Language Models as judges (LLM-as-a-Judge). We introduce a formal procedure for score normalization, reliability-weighted aggregation, and inter-annotator agreement (IAA) measurement to mitigate single-judge bias and enhance evaluation stability. Performance is quantified across three dimensions—quality, latency, and computational cost—enabling Pareto frontier analysis for a holistic understanding of system trade-offs.

## Core Methodology

Our framework is built upon a rigorous, academic-level methodology to ensure the production of verifiable and generalizable scientific claims.

### 1. Problem Formulation

The primary objective is to evaluate a multimodal QA system, denoted as a function $f$, which generates an answer $a$ given a query $q$ and contextual inputs from a document page.

-   **Input**: A document page represented by an image $I$, its OCR text $T$, and captions for visual elements $C$.
-   **Query**: A question $q_i$ from an automatically generated set $Q_p = \{q_1, ..., q_N\}$ for page $p$.
-   **System Under Test (SUT)**: The function $f$ is evaluated under $M$ different input modality combinations, $C^{(m)}$.
-   **Output**: An answer $a_i^{(m)} = f(q_i, C^{(m)})$, which is the subject of our evaluation.

### 2. Input Modality Combinations

To systematically analyze the contribution of each input modality, we define a fixed set of combinations for ablation studies:

-   **M1**: Image-only: $C^{(1)} = \{I\}$
-   **M2**: Image + Caption: $C^{(2)} = \{I, C\}$
-   **M3**: Image + OCR Text: $C^{(3)} = \{I, T\}$
-   **M4**: Image + Caption + OCR Text: $C^{(4)} = \{I, C, T\}$
-   **M5**: Cropped Regions + Caption + OCR Text: $C^{(5)} = \{\{I_r\}, C, T\}$
-   **M6**: Text-only (OCR + Caption): $C^{(6)} = \{T, C\}$
-   **M7**: Text-only (OCR): $C^{(7)} = \{T\}$

### 3. Automated Test Case Generation (`QueryGen`)

A dedicated language model (`LLM #1`) generates a standardized and scalable test set, operating exclusively on the textual context $\{T, C\}$ to remain blind to image pixels and ground-truth answers.

### 4. Multi-Stage Evaluation Protocol

Each generated answer undergoes a two-stage evaluation:

1.  **Stage 1: Heuristic Filtering**: A lightweight, rule-based procedure provides an initial quality signal by checking for **Grounding Validity**, **Numerical Consistency**, and **Format Compliance**.

2.  **Stage 2: Multi-Judge LLM Ensemble Evaluation**: A panel of distinct LLMs (e.g., GPT-4o, Gemini 1.5 Pro, Claude 3 Sonnet) independently assesses each answer. To mitigate bias, the identity of the system is anonymized, and all judges use a standardized rubric to score answers on **Accuracy**, **Groundedness**, **Completeness**, and **Clarity**.

    Scores from the judge panel are aggregated using a **reliability-weighted ensemble** approach after robust z-score normalization to produce a single, stable final score $S^{\star}$.

### 5. Quantitative Metrics & Analysis

We collect metrics across three categories:

1.  **Quality**: The final ensemble score $S^{\star}$.
2.  **Latency**: Wall-clock time (ms) for response generation.
3.  **Cost**: Computational cost in USD per query.

Statistical significance is determined using the **Wilcoxon signed-rank test** with **Holm-Bonferroni correction** for multiple comparisons. Results are synthesized in **Pareto frontier plots** to visualize quality-efficiency trade-offs.



---

## Features

-   **Advanced PDF Processing**: Uses MinerU for layout detection and content extraction.
-   **Multi-Modal Analysis**: Combines text, layout, and image processing for comprehensive document understanding.
-   **Automated Question Generation**: Generates structured, diverse questions from document content.
-   **Controlled Answer Generation**: Produces answers using various VLM models with multiple combination strategies.
-   **Robust Evaluation Pipeline**: Judges answer quality with heuristic and model-based evaluation using a multi-judge ensemble.
-   **Asynchronous Processing**: Handles large-scale document processing efficiently with sophisticated retry mechanisms.

## Project Structure

```

RiQF-eval/
├── .env                  \# Environment variables (API keys)
├── README.md             \# This file
├── README-ko.md          \# Korean version of the README
├── requirements.txt      \# Project dependencies
├── data/                 \# Input documents (e.g., test\_1.pdf)
├── notebooks/            \# Jupyter notebooks for analysis
├── results/              \# Output directory for all processing stages
│   └── [document\_name]/
│       ├── pages/
│       ├── mineru\_raw\_output/
│       ├── preprocessed\_context/
│       ├── generated\_questions/
│       └── evaluation\_results/
└── src/                  \# Source code
├── main.py           \# Main orchestration script
├── preprocess.py     \# Document preprocessing with MinerU
├── query\_gen.py      \# Question generation module
├── answer\_gen.py     \# Answer generation module
├── judge.py          \# Answer evaluation and judging
└── analysis.py       \# Result analysis and metrics

````

## Installation

### Prerequisites

1.  **Python Environment**: Python 3.8+
2.  **MinerU**: Document parsing engine.
3.  **Poppler**: A PDF rendering library for PDF-to-image conversion.
4.  **API Keys**: OpenAI, Anthropic, or Google AI API keys for model access.

### Setup

1.  **Install Dependencies**:
    ```bash
    # Install the uv package manager
    pip install uv

    # Install core MinerU dependencies
    uv pip install "mineru[core]"

    # Install additional requirements
    pip install -r requirements.txt
    ```

2.  **Install System Dependencies**:
    ```bash
    # macOS (using Homebrew)
    brew install poppler

    # Ubuntu/Debian
    sudo apt-get install -y poppler-utils
    ```

3.  **Environment Configuration**: Create a `.env` file in the project root and add your API keys.
    ```bash
    # Create .env file with your API key(s)
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    ```

## Usage

### Command Line Interface

The main script supports comprehensive document processing through various flags:

```bash
python src/main.py [OPTIONS]
````

### Command Line Arguments

| Argument               | Type  | Default                                            | Description                                          |
| ---------------------- | ----- | -------------------------------------------------- | ---------------------------------------------------- |
| `--input`              | str   | `"data/"`                                          | Path to the file or folder to process.               |
| `--output`             | str   | `"results/"`                                       | Output directory for all results.                    |
| `--concurrency`        | int   | `5`                                                | Maximum number of concurrent asynchronous tasks.     |
| `--interval`           | float | `1.5`                                              | Minimum wait time between API requests (seconds).    |
| `--save_context`       | flag  | `False`                                            | Save preprocessed page contexts as JSON files.       |
| `--captioning`         | flag  | `False`                                            | Enable image captioning with a VLM.                  |
| `--generate_questions` | flag  | `False`                                            | Generate questions from the document content.        |
| `--generate_answers`   | flag  | `False`                                            | Generate answers for the questions.                  |
| `--judge_answers`      | flag  | `False`                                            | Evaluate the quality of the generated answers.       |
| `--models`             | list  | `["gpt-4o"]`                                       | Models to use for answer generation.                 |
| `--combos`             | list  | `["M1", "M2", ...]`                                | Input modality combination strategies to test.       |
| `--judges`             | list  | `["gpt-4o", "gemini-1.5-pro", "claude-3-sonnet"]` | Judge models for the evaluation ensemble.            |
| `--run_analysis`       | flag  | `False`                                            | Run the final analysis and reporting stage.          |

### Usage Examples

#### Full Pipeline on a Single Document

```bash
python src/main.py \
    --input data/test_1.pdf \
    --output results/test_run \
    --save_context \
    --captioning \
    --generate_questions \
    --generate_answers \
    --judge_answers \
    --run_analysis
```

#### Batch Processing with Custom Models

```bash
python src/main.py \
    --input data/ \
    --output results/batch_custom \
    --concurrency 4 \
    --interval 2.0 \
    --models gpt-4-turbo gpt-4o \
    --combos M1 M4 M7 \
    --judges gpt-4o \
    --generate_answers \
    --judge_answers
```

## Performance Optimization

The system includes sophisticated rate limiting and retry mechanisms to handle large-scale processing and API limitations gracefully.

  - **Rate Limiting**: Configurable concurrency levels and interval-based request spacing.
  - **Retry Mechanism**: Implements exponential backoff for up to 5 attempts on failed operations, with intelligent waiting to avoid rate limits.
  - **Memory Management**: Asynchronous processing and streaming JSONL outputs are used to handle large documents and result sets efficiently.

## Troubleshooting

1.  **Rate Limiting Errors**: Reduce concurrency and increase the interval.
    `--concurrency 2 --interval 3.0`
2.  **API Key Issues**: Verify that your `.env` file is correctly formatted and located in the project root.
3.  **Memory Issues**: For very large documents, process with lower concurrency.
    `--concurrency 1`

## Citation

If you use this framework in your research, please cite the following authors:

  - **Jimin Park**: xistoh162108@kaist.ac.kr, xistoh162108@hyper-x.ai
      - Development Intern, HyperX Inc.
      - Undergraduate Student, KAIST School of Computing
  - **Jongrye Jang**: cto@hyper-x.ai
      - Chief Technology Officer (CTO), HyperX Inc.  
  - **Hanbin Ji**: ceo@hyper-x.ai
      - Chief Executive Officer (CEO), HyperX Inc.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
