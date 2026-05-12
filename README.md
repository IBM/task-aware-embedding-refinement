<div align="center">

# 🚀 LLM-Guided Embedding Refinement

**Boost zero-shot classification and retrieval with test-time query optimization**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
</div>

---

## 📖 Overview

This repository contains code to improve **zero-shot classification and retrieval using embedding models** through test-time optimization of query embedding representations.

At test time, given a user query, the query embedding is optimized with gradient descent based on targeted feedback from a stronger model. The method uses scores from an LLM or reranker over a small sampled set of candidate documents, then updates the query representation so the embedding space better reflects the task-specific intent of the query.

### 🎯 Key Features

- ✨ **Test-time optimization** — No retraining required
- 🔄 **Flexible architecture** — Works with various text embedding models and LLM rerankers
- 📊 **Proven results** — Consistent gains across multiple benchmarks

### 📄 Paper

> **Task-Aware Embedding Refinement via Test-time LLM Guidance**  
> *Ariel Gera, Shir Ashury-Tahan, Gal Bloch, Ohad Eytan, Assaf Toledo*

---

## 💡 How It Works

Embedding models are efficient and scalable, but in challenging zero-shot settings they may miss nuanced task constraints. This library explores a test-time refinement procedure that adapts the query representation using external guidance from a generative LLM, **without retraining the embedding model**.

This approach improves ranking quality across multiple search and classification benchmarks, with consistent gains on tasks such as:
- 📚 Literature search
- 🎯 Intent detection
- 🔑 Key-point matching
- 📋 Instruction-following retrieval

### 🔄 Workflow
**Step-by-step process:**

1. 📝 Embed the original query and candidate documents
2. 🔍 Retrieve top candidates by embedding similarity
3. 🎯 Score a sampled subset using an LLM, cross-encoder reranker, or gold labels
4. ⚡ Optimize the query embedding to better align with supervision
5. 🔄 Re-score the corpus using the refined query embedding

---

## 🚀 Quick Start

### 📦 Installation

#### 1️⃣ Create and activate a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

#### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

#### 3️⃣ Configure inference for LiteLLM or OpenAI

The reranker and [HyDE](https://github.com/texttron/hyde) components use an OpenAI-compatible chat-completions API.

**Option A — LiteLLM gateway or proxy**

Set `BASE_URL` to your LiteLLM endpoint and `API_KEY` to the corresponding key:

```bash
export BASE_URL="http://localhost:4000"
export API_KEY="your-litellm-api-key"
```

Use LiteLLM explicitly by prefixing the model name with `LiteLLM/`, for example:

```bash
python run_experiments.py \
  --reranker_model "LiteLLM/mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
```

**Option B — OpenAI API**

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Use OpenAI explicitly by prefixing the model name with `OpenAI/`, for example:

```bash
python run_experiments.py \
  --reranker_model "OpenAI/gpt-4.1-mini" \
  --hyde_model "OpenAI/gpt-4.1-mini"
```

**Important notes**
- If no service prefix is provided, this repository defaults to **LiteLLM** for LLM inference. This will fail if you did not define a suitable endpoint in your environment.
- `--reranker_model` controls the LLM used for relevance feedback during optimization.
- `--hyde_model` controls the LLM used to generate hypothetical documents for HyDE (HYpothetical Document Embeddings, see [here](https://github.com/texttron/hyde)), this is optional and is not required for basic query optimization functionality.
- LiteLLM/OpenAI setup is only required when using LLM-based reranking or HyDE. It is not needed for `--optimize_with_gold` or cross-encoder rerankers such as `cross-encoder/ms-marco-MiniLM-L-6-v2`.

### ▶️ Basic Usage

**Run all default models on all datasets:**
```bash
python run_experiments.py
```

**Run a specific model on selected datasets:**
```bash
python run_experiments.py \
  --embedding_models "Qwen/Qwen3-Embedding-0.6B" \
  --datasets "Clinc150" "NFCorpus"
```

**Run experiments in parallel:**
```bash
python run_experiments.py --parallel 3
```

---

## 📊 Supported Datasets

The repository currently supports the following datasets through `dataset_loaders.py`:

| Dataset | Description | Reference |
|---------|-------------|-----------|
| 🎓 **RealScholarQuery** | Real-world academic search queries over arXiv CS papers | [He et al., 2025](https://aclanthology.org/2025.acl-long.572/) |
| 🔑 **KPM** | Key-point matching from 2021 KPA shared task | [Friedman et al., 2021](https://aclanthology.org/2021.argmining-1.16/) |
| 📋 **FollowIR** | Information retrieval from TREC relevance narratives | [Weller et al., 2025](https://aclanthology.org/2025.naacl-long.597/) |
| 💬 **Clinc150** | Intent classification with 150 intents across 10 domains | [Larson et al., 2019](https://arxiv.org/abs/1909.02027) |
| 🏦 **Banking77** | Banking domain with 77 fine-grained intent categories | [Casanueva et al., 2020](https://aclanthology.org/2020.nlp4convai-1.5/) |
| 🏥 **NFCorpus** | Medical literature retrieval with lay queries | [Boteva et al., 2016](https://doi.org/10.1007/978-3-319-30671-1_58) |

---

## 🛠️ Custom Usage

### 🎮 Main Entry Points

| Script | Purpose |
|--------|---------|
| `embedding_adaptation.py` | Core script for single experiment runs |
| `run_experiments.py` | Batch runner for multiple experiments |

### 🔧 Command Examples

<details>
<summary><b>Preview commands without execution</b></summary>

```bash
python run_experiments.py --dry_run
```
</details>

<details>
<summary><b>Enable HyDE (Hypothetical Document Embeddings)</b></summary>

```bash
python run_experiments.py \
  --hyde_model "meta-llama/Llama-3.1-8B-Instruct"
```
</details>

<details>
<summary><b>Run single experiment with custom parameters</b></summary>

```bash
python embedding_adaptation.py \
  --embedding_model "Qwen/Qwen3-Embedding-0.6B" \
  --dataset "NFCorpus" \
  --reranker_model "mistralai/Mistral-Small-3.2-24B-Instruct-2506" \
  --lr 1e-4 \
  --num_steps 100 \
  --total_scores 20 \
  --scores_from_top 20
```
</details>

### ⚙️ Configuration Parameters

#### 🎯 Single Experiment Parameters (`embedding_adaptation.py`)

These parameters configure individual experiment runs. They can also be passed to `run_experiments.py` and will be forwarded to each experiment.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--embedding_model` | Embedding model to use (single model) | `Qwen/Qwen3-Embedding-0.6B` |
| `--dataset` | Dataset to use for experiment | `RealScholarQuery` |
| `--reranker_model` | Reranker model for feedback | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` |
| `--hyde_model` | LLM for hypothetical document generation (optional) | None |
| `--lr` | Learning rate for query embedding optimization | 1e-4 |
| `--num_steps` | Number of optimization steps | 100 |
| `--total_scores` | Total documents to sample for reranking signal | 20 |
| `--scores_from_top` | Documents sampled from top results (by embedding similarity) | 20 |
| `--optimize_with_gold` | Use gold labels instead of reranker scores | False |
| `--embedder_batch_size` | Batch size for embedding inference | 10 |
| `--reranker_batch_size` | Batch size for reranker inference | 10 |
| `--random_seed` | Random seed for reproducibility | 42 |
| `--save_tensors` | Save query trajectory tensors for analysis | False |
| `--experiment_name` | Custom experiment name (auto-generated if not provided) | None |

#### 🔄 Batch Runner Parameters (`run_experiments.py` only)

These parameters are specific to the batch runner and control how multiple experiments are executed.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--parallel` | Number of concurrent experiments to run | 1 |
| `--experiment_prefix` | Custom prefix for experiment names | None |
| `--embedding_models` | List of embedding models to test (space-separated) | See defaults in script |
| `--datasets` | List of datasets to test (space-separated) | See defaults in script |
| `--dry_run` | Preview commands without executing | False |
| `--continue_on_error` | Continue running experiments even if one fails | False |

---

## 📁 Output Structure

Each experiment run creates a directory under `output/<experiment_name>/` containing:

```
output/
└── <experiment_name>/
    ├── *_results.csv              # Per-topic evaluation metrics
    ├── *_raw_scores.parquet       # Raw document-level scores
    ├── tensors/                   # Query trajectory tensors (if enabled)
    └── config.json                # Experiment configuration metadata
```

---

## ⏱️ Runtime Estimates

The total runtime of the experiments consists of two main components:

1. **Document embeddings** — Embedding all documents in the corpus. In a deployment environment this would typically be done offline.
2. **Per-query computations** — Query embedding, refinement optimization, and LLM teacher feedback.

Per-query latency with a GPU is well under a second per query, as shown in the paper. This means just a few minutes to run all the queries in a dataset, assuming an efficient model endpoint for obtaining the LLM feedback scores.

Thus, much of the experiment runtime is devoted to the one-time cost of computing the corpus document embeddings. For convenience, it is possible to run this initial step separately using the script `embed_all_documents.py`.

**Runtime estimates on a single A100-80GB GPU:**

- Using a **small embedding model**, like *Qwen/Qwen3-Embedding-0.6B*: 
Running the full experiment on **all datasets**, including embedding all corpus documents and optimizing all the queries, should take about an hour.

- Using **larger embedding models**, like *Qwen/Qwen3-Embedding-8B*:
  - Embedding corpus documents with 7B/8B models can range from a few minutes for small datasets (e.g., *KPM*) to 3-4 hours for larger datasets like *RealScholarQuery* or *FollowIR*.
  - Note that for datasets with long documents (e.g., FollowIR), it may be necessary to use a small `--embedder_batch_size` to avoid running out of GPU memory.

---

## 📝 Notes

> **💾 Caching:** Embeddings, reranker outputs, and generated texts are cached under `/cache` to avoid repeated computation.

> **🔬 Research Focus:** This implementation is optimized for research and experimentation. For production deployment, consider replacing the file-system cache with a scalable vector-store solution.

---

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{gera2026taskaware,
  title={Task-Aware Embedding Refinement via Test-time LLM Guidance},
  author={Gera, Ariel and Ashury-Tahan, Shir and Bloch, Gal and Eytan, Ohad and Toledo, Assaf},
  year={2026}
}
```
