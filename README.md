# Agentic RAG & Domain-Adapted LLM Fine-Tuning — JPMorgan Investment Research

**Employer:** JPMorgan Chase | AI/ML Research & Investment Intelligence  
**Role:** Applied AI/ML Engineer  
**Timeline:** September 2024 – February 2025  
**Domain:** LLM Fine-Tuning · PEFT/LoRA · RAGAS Evaluation · Contract Intelligence · Agentic AI

---

## Overview

Two parallel engineering tracks addressing a failure mode identified during LLM Suite production: general-purpose LLMs produced systematically incomplete extraction of contractual obligation clauses despite the RAG pipeline correctly retrieving the relevant passages.

**Fine-tuning track:** Fine-tuned Mistral-7B using LoRA on JPMorgan's internal legal contract corpus, achieving obligation extraction F1 of 0.92 — a 51% relative improvement over zero-shot baseline and outperforming GPT-4 few-shot (F1: 0.84) at 87% lower inference cost.

**Agentic track (contributor):** Multi-agent investment research assistant with specialized sub-agents for structured data, unstructured document retrieval, and proprietary analytics.

I held primary ownership of: SFT dataset preparation, RAGAS evaluation harness, BERTScore integration, FastAPI inference wrapper.

---

## Problem Statement

JPMorgan's compliance team used the LLM Suite RAG pipeline to query financial contracts for obligation data. Production evaluation revealed a consistent failure:

- RAG retrieval layer correctly surfaced relevant contract passages (high precision)
- LLM generation layer produced **structurally incomplete** obligation extractions:
  - Obligated party: correctly identified in 88% of cases ✓
  - Obligation type: correctly identified in 88% of cases ✓
  - Full deadline expression: correctly captured in only **61%** of cases ✗
  - Complete consequence-of-non-performance clause: only **54%** of cases ✗

These weren't hallucinations — the model wasn't inventing incorrect information. It was systematically omitting material components of legal obligations. For a compliance team monitoring covenant violations, an incomplete obligation is operationally unacceptable.

**Root cause:** General-purpose LLMs lack exposure to financial legal contract syntax. Constructs like *"the Obligor shall, no later than the third Business Day following the Trigger Date, deliver to the Counterparty a Compliance Certificate in the form attached as Schedule 4"* require domain-conditioned parsing that zero-shot prompting cannot reliably achieve at scale.

**Strategic constraint:** Running Mistral-7B on JPMorgan's internal GCP infrastructure meant contract data never left the bank's controlled environment — a non-negotiable requirement for contracts containing material non-public information.

---

## My Contributions

| Workstream | Role | Ownership |
|-----------|------|-----------|
| SFT dataset preparation (3,200 segments) | Dataset Specialist | **Primary** |
| NLTK sliding window clause segmentation | Builder | **Primary** |
| spaCy NER initial annotation pipeline | Builder | **Primary** |
| LoRA rank ablation experiments (r=4,8,16) | Training Contributor | Contributor |
| MLflow training monitoring | Builder | **Primary** |
| RAGAS evaluation harness (4 dimensions) | Evaluation Engineer | **Primary** |
| BERTScore integration (DeBERTa-XL) | Builder | **Primary** |
| 200-pair gold-standard QA dataset | Builder | **Primary** |
| FastAPI inference wrapper (API parity) | Builder | **Primary** |
| Latency benchmarking | Builder | **Primary** |
| MRM model card (primary author) | Author | **Primary** |
| GCP Vertex AI training infrastructure | Contributor | Contributor |

---

## Technical Architecture

```
Raw Contract Corpus (3,200 segments)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  DATASET PREPARATION  (Chandan — Primary)            │
│  spaCy NER pipeline → initial entity annotations    │
│  NLTK sliding window (3-sentence, stride 1)          │
│  Variable window extension for termination clauses   │
│  (22 keyword triggers → 5-sentence window)           │
│  Pandas stratification: 75/15/10 splits              │
│  Output: JSONL instruction-response pairs            │
│  Legal reviewer validation: 11% segments relabeled   │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  LORA FINE-TUNING  (Contributor)                     │
│  Base: Mistral-7B-Instruct-v0.2                      │
│  PEFT: LoRA r=16, alpha=32, dropout=0.05             │
│  Targets: q_proj, v_proj (attention matrices only)   │
│  Trainable params: 4.2M of 7B (0.06%)               │
│  Training: GCP Vertex AI, 4x A100 40GB, bfloat16    │
│  MLflow: 22 runs, early stop at epoch 2.4            │
│  Result: Obligation F1 0.92 (+51% vs zero-shot)     │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  RAGAS EVALUATION HARNESS  (Chandan — Primary)       │
│  200-pair gold-standard QA dataset                   │
│  4 dimensions: faithfulness · relevancy ·            │
│    context precision · context recall                 │
│  BERTScore (DeBERTa-XL) for consequence clauses      │
│  Human annotations for primary faithfulness metric   │
│  (avoids GPT-4 judge bias)                           │
│  Result: Faithfulness 0.91 vs GPT-4 few-shot 0.93   │
└────────────────────────────┬─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  FASTAPI INFERENCE WRAPPER  (Chandan — Primary)      │
│  GCP Vertex AI Prediction endpoint                   │
│  API contract parity with GPT-4 pipeline             │
│  Zero compliance team integration changes required   │
│  Configurable model_provider metadata override       │
│  Latency: 1.4s/seg vs 3.8s GPT-4 (63% reduction)   │
└──────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

### Why LoRA Over Full Fine-Tuning
Full fine-tuning Mistral-7B = updating 7B parameters. On a 3,200-sample corpus, this risks catastrophic forgetting and overfitting. LoRA freezes base weights and injects low-rank adapter matrices into attention layers: **4.2M trainable parameters (0.06% of total)**. The adapter is also independently swappable — the compliance team can use different specializations for different contract types without retraining the base model.

### Variable-Window Segmentation for Termination Clauses
Standard 3-sentence sliding window fragmented multi-sentence termination clauses (which enumerate 5–8 triggering conditions). Solution: if a candidate segment contains any of 22 identified termination trigger keywords, extend window to 5 sentences. This improved termination clause coverage without inflating average context length.

### Human-Annotated Triples for Primary Faithfulness Metric
RAGAS uses GPT-4 as judge for faithfulness scoring. Problem: GPT-4 was evaluating a model designed to outperform GPT-4 on this specific task — a circular reasoning risk. Solution: construct 200 human-annotated triples for the primary metric; use GPT-4 as judge only for additional non-annotated coverage, reported separately. MRM reviewer specifically cited this as a methodological strength.

### API Contract Parity via Configurable Metadata Override
The compliance team's audit trail system used `model_provider` field as a filter key for regulatory reporting. Switching from GPT-4 to Mistral-7B required changing this field value. Rather than requiring the compliance team to update their logging schema, added a configurable response metadata override in the FastAPI wrapper. This became a standard pattern for future model backend substitutions.

---

## Results

### Fine-Tuning Performance
| Model | Obligation F1 | RAGAS Faithfulness | Cost/1K segments |
|-------|--------------|-------------------|-----------------|
| Base Mistral-7B (zero-shot) | 0.61 | - | Low |
| GPT-4 (zero-shot) | 0.79 | 0.89 | High |
| GPT-4 (few-shot) | 0.84 | 0.93 | High |
| **Fine-tuned Mistral-7B** | **0.92** | **0.91** | **87% lower than GPT-4** |

### BERTScore vs. Exact-Match on Consequence Clauses
| Metric | Score |
|--------|-------|
| Exact-match F1 | 0.71 |
| BERTScore F1 (DeBERTa-XL) | **0.89** |

BERTScore revealed an 18-point gap — the binary metric was systematically underscoring partially correct extractions that were semantically accurate.

### Inference Performance
| Metric | Fine-tuned Mistral-7B | GPT-4 API |
|--------|----------------------|-----------|
| Mean latency/segment | **1.4s** | 3.8s |
| Cost/1K segments | **87% lower** | baseline |
| Latency reduction | **63%** | - |

### Dataset & Evaluation
- 3,200-segment SFT dataset, 75/15/10 splits, 4 obligation categories
- 11% segments relabeled by legal reviewers (obligation type field)
- 200-pair gold-standard RAGAS evaluation dataset (first human-annotated LLM eval corpus for this use case at JPMorgan)
- 22 MLflow experiment runs, full LoRA rank ablation provenance

---

## Stack

| Layer | Technology |
|-------|-----------|
| Fine-Tuning | Hugging Face PEFT (LoRA r=16), TRL (SFT), Mistral-7B-Instruct-v0.2, bfloat16 |
| Training Infra | GCP Vertex AI, 4x NVIDIA A100 40GB, gradient checkpointing |
| Dataset Prep | spaCy NER pipeline, NLTK (sliding window), Pandas, JSONL |
| Evaluation | RAGAS (4 dimensions), BERTScore (DeBERTa-XL) |
| Experiment Tracking | MLflow (22 runs, LoRA adapter checkpoints, per-class F1 logging) |
| Serving | FastAPI, GCP Vertex AI Prediction, Pydantic schemas |

---

## Project Structure

```
3_Agentic_RAG_LLM_FineTuning/
├── README.md
├── dataset_preparation/
│   ├── sft_dataset_builder.py       # spaCy NER annotation + NLTK sliding window segmentation
│   ├── sliding_window_segmenter.py  # Variable-window extension for termination clauses
│   ├── stratified_splitter.py       # Pandas stratification by obligation type
│   └── jsonl_formatter.py           # Instruction-response pair formatting
├── fine_tuning/
│   ├── lora_trainer.py              # LoRA fine-tuning with Hugging Face PEFT + TRL
│   ├── rank_ablation.py             # Ablation study: r=4, r=8, r=16
│   └── mlflow_tracking.py           # 22-run experiment tracking
├── evaluation/
│   ├── ragas_harness.py             # RAGAS 4-dimension evaluation pipeline
│   ├── bertscore_integration.py     # DeBERTa-XL semantic similarity scorer
│   ├── gold_dataset_builder.py      # 200-pair QA dataset construction
│   └── benchmark_report.py          # Comparative benchmark: Mistral vs GPT-4
├── serving/
│   ├── fastapi_wrapper.py           # REST API wrapping GCP Vertex AI endpoint
│   ├── schemas.py                   # Pydantic schemas (API contract parity with GPT-4)
│   └── latency_benchmark.py         # Concurrent load benchmarking
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_lora_ablation_results.ipynb
│   ├── 03_ragas_evaluation.ipynb
│   └── 04_bertscore_analysis.ipynb
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Setup & Usage

```bash
pip install -r requirements.txt

# Build SFT dataset from raw contract segments
python dataset_preparation/sft_dataset_builder.py \
    --input_dir data/raw_contracts \
    --output_dir data/sft_dataset \
    --n_segments 3200

# Run LoRA rank ablation
python fine_tuning/rank_ablation.py \
    --dataset data/sft_dataset \
    --ranks 4 8 16 \
    --mlflow_experiment lora_ablation

# Run RAGAS evaluation
python evaluation/ragas_harness.py \
    --model_endpoint your_vertex_ai_endpoint \
    --dataset evaluation/gold_qa_dataset.jsonl \
    --output evaluation/results.json

# Start inference API
uvicorn serving.fastapi_wrapper:app --port 8001
```

---

## Key Learnings

1. **LoRA's parameter efficiency matters most on small fine-tuning corpora.** 3,200 segments is small by LLM standards. Training only 4.2M parameters inherently limits overfitting capacity — this was the right architectural choice, not a compromise.

2. **Don't use the same model to evaluate the model it's competing against.** Using GPT-4 to judge faithfulness of a model designed to outperform GPT-4 creates circular reasoning. Human-annotated triples for the primary metric, GPT-4 as judge only for secondary coverage.

3. **BERTScore reveals what exact-match misses.** Consequence clause exact-match F1 of 0.71 suggested poor performance. BERTScore F1 of 0.89 revealed the model was semantically accurate — it just used different wording. Always check semantic similarity for free-text extraction tasks.

4. **API contract parity is the hidden integration work.** The FastAPI wrapper's `model_provider` metadata field was a one-line code change that prevented the compliance team from needing to update their regulatory reporting pipeline. Small design choices during API design have outsized downstream impact.
