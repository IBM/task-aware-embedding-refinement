import argparse
import ast
import json
import os
import random
import time
import statistics as stats
from collections import defaultdict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from typing import Any

from dataset_loaders import Datasets
from embedders import get_embedder, CACHE_DIR
from rerankers import get_reranker, CACHE_DIR as RERANKER_CACHE_DIR
from utils import load_instruction_templates, get_templates_for_model, get_llm_task_instruction, apply_template, get_device
from embedding_adaptation import optimize


def sync_accel():
    """Synchronize accelerator to ensure accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


class BenchRunner:
    """Benchmarking runner that measures execution time with warmup."""
    def __init__(self):
        self.results = []

    def bench(self, name, fn, *, group=None, dataset_id=None):
        # Warmup:
        fn()
        sync_accel()
        # Measure
        t0 = time.perf_counter()
        fn()
        sync_accel()
        dt = (time.perf_counter() - t0) * 1000  # Convert to milliseconds
        r = {
            "name": name,
            "group": group,
            "dataset": dataset_id,
            "values": [dt],
            "mean": dt,
        }
        self.results.append(r)
        return r

    def aggregate(self, replace=False):
        by_group = defaultdict(list)
        for r in self.results:
            g = r.get("group")
            if g:
                by_group[g].extend(r["values"])
        aggs = []
        for g, vals in by_group.items():
            aggs.append({
                "name": g,
                "values": vals,
                "mean": stats.fmean(vals) if vals else None,
                "median": stats.median(vals) if vals else None,
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "stdev": stats.pstdev(vals) if len(vals) > 1 else 0.0,
            })
        if replace:
            self.results = aggs
        else:
            self.results.extend(aggs)

    def dump(self, path, extra_meta=None):
        out = {"benchmarks": self.results, "metadata": extra_meta or {}}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure latencies of embedding adaptation components"
    )

    parser.add_argument(
        "--optimize_with_gold",
        type=ast.literal_eval,
        default=False,
        help="Use gold instead of reranker feedback for optimization"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default="mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        help="Name of the reranker model to use"
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=10,
        help="Batch size for reranker inference"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="RealScholarQuery",
        choices=Datasets.all_datasets(),
        help="Dataset to use for experiments"
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=50,
        help="Number of random queries to sample for latency measurements"
    )

    # Optimization parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for query embedding optimization"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of optimization steps"
    )

    # Sampling parameters
    parser.add_argument(
        "--total_scores",
        type=int,
        default=20,
        help="Total number of documents to sample for reranking signal"
    )
    parser.add_argument(
        "--scores_from_top",
        type=int,
        default=20,
        help="Number of top documents (by embedding similarity) to include in sampling"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="latency_results",
        help="Output directory for results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Get dataset
    dataset_cls = getattr(Datasets, args.dataset)
    query_to_label_series = dataset_cls().load_gold()
    all_queries = sorted(query_to_label_series.keys())
    
    # Sample random queries
    sampled_queries = random.sample(all_queries, min(args.num_queries, len(all_queries)))
    print(f"Sampled {len(sampled_queries)} queries from {args.dataset}")
    
    # Load instruction templates
    templates_yaml_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "instruction_templates.yaml")
    instruction_templates = load_instruction_templates(templates_yaml_path)
    query_template, document_template = get_templates_for_model(
        args.embedding_model,
        instruction_templates,
        dataset_name=args.dataset
    )
    
    # Initialize embedder with cache for document embeddings
    embedding_model = get_embedder(
        args.embedding_model,
        cache_dir=os.path.join(CACHE_DIR, dataset_cls.dataset_name)
    )
    
    # Create a direct embedder instance (without cache) for benchmarking query embeddings
    embedder_instance = embedding_model.embedder_cls()
    direct_embedder: Any = embedder_instance.create_embedder(args.embedding_model)
    
    # Initialize reranker if not using gold labels
    if not args.optimize_with_gold:
        llm_task_instruction = get_llm_task_instruction(instruction_templates, args.dataset)
        reranker_cache_dir = os.path.join(RERANKER_CACHE_DIR, dataset_cls.dataset_name)
        reranker = get_reranker(
            args.reranker_model,
            cache_dir=reranker_cache_dir,
            task_instruction=llm_task_instruction
        )
        print(f"Initialized reranker: {args.reranker_model}")
        if llm_task_instruction:
            print(f"  with task instruction: {llm_task_instruction}")
    else:
        reranker = None
    
    # Setup output directory
    out_dir = os.path.join("output", args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    runner = BenchRunner()
    
    meta = {
        "device": str(device),
        "torch": torch.__version__,
        "cuda": getattr(torch.version, "cuda", None),
        "mps": torch.backends.mps.is_available(),
        "embedding_model": args.embedding_model,
        "dataset": args.dataset,
        "num_queries": len(sampled_queries),
        "num_steps": args.num_steps,
        "lr": args.lr,
        "total_scores": args.total_scores,
        "scores_from_top": args.scores_from_top,
    }
    
    print("\n" + "="*80)
    print("Starting benchmarks...")
    print("="*80 + "\n")
    
    # Benchmark each sampled query
    for query_idx, query in enumerate(tqdm(sampled_queries, desc="Benchmarking queries")):
        topic_gold_series = query_to_label_series[query]
        topic_gold_series = topic_gold_series[~topic_gold_series.index.duplicated(keep='first')]
        
        # Get documents for this query
        documents = topic_gold_series.index.tolist()
        
        # 1. Benchmark query embedding (without cache)
        templated_query = apply_template([query], query_template)
        runner.bench(
            f"embed_query[{query_idx}]",
            partial(direct_embedder.embed_documents, templated_query),
            group="embed_query",
            dataset_id=args.dataset,
        )
        query_embedding = direct_embedder.embed_documents(templated_query)[0]
        query_tensor = torch.tensor(query_embedding, device=device)
        
        # Embed documents (not benchmarked, just for setup)
        templated_docs = apply_template(documents, document_template)
        doc_embeddings = embedding_model.embed_documents(templated_docs, batch_size=10)
        doc_embeddings_tensor = torch.tensor(doc_embeddings, device=device)
        
        # 2. Benchmark similarity calculation
        def calc_similarity():
            return F.cosine_similarity(doc_embeddings_tensor, query_tensor.detach())
        
        runner.bench(
            f"calc_similarity[{query_idx}]",
            calc_similarity,
            group="calc_similarity",
            dataset_id=args.dataset,
        )
        similarities = calc_similarity().cpu()
        
        # 3. Benchmark optimization using the original optimize function
        def run_optimize():
            return optimize(
                doc_embeddings_tensor=doc_embeddings_tensor,
                topic_gold_series=topic_gold_series,
                similarities=similarities.to(device),
                query_tensor=query_tensor,
                lr=args.lr,
                num_steps=args.num_steps,
                total_scores=args.total_scores,
                scores_from_top=args.scores_from_top,
                random_seed=args.random_seed,
                reranker=reranker,
                reranker_batch_size=args.reranker_batch_size,
                topic=query,
                experiment_dir=None,
                optimize_with_gold=args.optimize_with_gold,
                save_tensors=False,
                refit=False,
            )
        
        runner.bench(
            f"optimize[{query_idx}]",
            run_optimize,
            group="optimize",
            dataset_id=args.dataset,
        )
    
    # Aggregate results
    print("\n" + "="*80)
    print("Aggregating results...")
    print("="*80 + "\n")
    
    runner.aggregate(replace=True)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    for result in runner.results:
        print(f"{result['name']:30s} | "
              f"mean: {result['mean']:8.2f}ms | "
              f"median: {result['median']:8.2f}ms | "
              f"min: {result['min']:8.2f}ms | "
              f"max: {result['max']:8.2f}ms | "
              f"stdev: {result['stdev']:8.2f}ms")
    
    # Save results
    out_json = os.path.join(out_dir, f"embedding_adaptation_latencies_{args.dataset}.json")
    runner.dump(out_json, extra_meta=meta)
    print(f"\n{'='*80}")
    print(f"Results saved to: {out_json}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
