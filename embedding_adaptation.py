import argparse
import ast
import os.path
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import tqdm
import torch
import torch.nn.functional as F
from unitxt.metrics import RocAuc, RetrievalAtK

from dataset_loaders import Datasets
from embedders import EmbedderWithFileSystemCache, get_embedder, CACHE_DIR
from rerankers import get_reranker, CACHE_DIR as RERANKER_CACHE_DIR
from generative_llms import get_llm, generate_hypothetical_documents, CACHE_DIR as LLM_CACHE_DIR
from utils import load_instruction_templates, get_templates_for_model, get_llm_task_instruction, apply_template, \
    save_experiment_config, get_device, slugify


def evaluate(gold_labels, predictions):
    """Evaluate retrieval performance using standard metrics."""
    auc_metric = RocAuc()
    retrieval_metric = RetrievalAtK(k_list=[5, 10, 15, 20, 30, 40, 50])

    gold_for_unitxt = [[g] for g in gold_labels]
    res_dict = auc_metric.compute(gold_for_unitxt, predictions, [{}])

    ref_ids_for_metric = [[i for i, gold in enumerate(gold_labels) if gold == 1]]
    predictions_for_metric = np.argsort(predictions)[::-1]
    res_dict.update(retrieval_metric.compute(ref_ids_for_metric, predictions_for_metric, {}))
    return res_dict


def kl_divergence(p, q, eps=1e-8):
    return torch.sum(p * torch.log((p + eps) / (q + eps)))


def _write_scores_batch(scores_list, output_path):
    """Write a batch of scores to parquet file, appending if file exists."""
    df_schema = {
        "topic": pl.Categorical,
        "document": pl.Categorical,
        "gold_label": pl.Int8,
        "score": pl.Float32,
        "reranker_score": pl.Float32,
    }
    batch_df = pl.DataFrame(scores_list, schema=df_schema)

    if os.path.exists(output_path):
        # Append to existing file
        existing_df = pl.read_parquet(output_path)
        combined_df = pl.concat([existing_df, batch_df])
        combined_df.write_parquet(
            output_path,
            compression="zstd",
            compression_level=9,
            use_pyarrow=False
        )
    else:
        # Create new file
        batch_df.write_parquet(
            output_path,
            compression="zstd",
            compression_level=9,
            use_pyarrow=False
        )


def save_query_updates(
    experiment_dir: str,
    topic: str,
    steps: list[int],
    loss_per_step: list[float],
    query_updates: list[torch.Tensor],
):
    topic_slug = slugify(topic)
    topic_dir = os.path.join(experiment_dir, "tensors", topic_slug)
    os.makedirs(topic_dir, exist_ok=True)

    torch.save(
        {
            "topic": topic,
            "steps": steps,  # includes 0 and final checkpoints
            "loss_per_step": torch.tensor(loss_per_step, dtype=torch.float32),
            "query_updates": torch.stack([t.to(torch.float32) for t in query_updates]),
        },
        os.path.join(topic_dir, "query_updates.pt"),
    )


def optimize(doc_embeddings_tensor, topic_gold_series, similarities, query_tensor, lr, num_steps,
             total_scores, scores_from_top, random_seed, reranker, reranker_batch_size, topic, experiment_dir,
             optimize_with_gold=False, save_tensors=False, refit=False):
    query_tensor = query_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([query_tensor], lr=lr)
    # sort everything according to the similarity scores
    sort_indices = torch.argsort(similarities, descending=True)
    sorted_similarities = similarities[sort_indices]
    sorted_doc_embeddings_tensor = doc_embeddings_tensor[sort_indices]
    sorted_topic_gold = topic_gold_series.iloc[sort_indices.cpu().numpy()]
    topic_sim_series = pd.Series(sorted_similarities.cpu().numpy(), index=sorted_topic_gold.index)

    indices_to_sample = list(range(scores_from_top))  # top docs by embeddings to get more relevant ones
    # add randomly sampled docs for diversity
    indices_to_sample += random.Random(random_seed).sample(range(scores_from_top, len(sorted_topic_gold)),
                                                           k=total_scores-scores_from_top)
    if optimize_with_gold:
        scores = [float(sorted_topic_gold.iloc[i]) for i in indices_to_sample]
    else:
        # get reranker scores for selected docs
        docs_to_rerank = [topic_sim_series.index[i] for i in indices_to_sample]
        scores = reranker.rerank(topic, docs_to_rerank, batch_size=reranker_batch_size)

    doc_to_reranker_score = {topic_sim_series.index[idx]: score
                             for idx, score in zip(indices_to_sample, scores)}

    if save_tensors:
        query_updates: list[torch.Tensor] = [query_tensor.detach().clone().cpu()]
        loss_per_step: list[float] = []
        sims_steps: list[int] = [0]

    sampled_docs_tensor = sorted_doc_embeddings_tensor[indices_to_sample].detach()
    s = torch.tensor(scores, device=query_tensor.device)
    if refit:
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        s = s / 2.0  # temperature T=2
    p = s.softmax(-1).detach()
    for step in tqdm.tqdm(range(num_steps)):
        q = F.cosine_similarity(sampled_docs_tensor, query_tensor)
        if refit:
            q = (q - q.min()) / (q.max() - q.min() + 1e-8)
        q = q.softmax(-1)
        loss = kl_divergence(p, q)
        optimizer.zero_grad(set_to_none=True)  # ensures the backward pass is only on params with a gradient
        loss.backward()
        optimizer.step()

        if save_tensors:
            loss_per_step.append(float(loss.detach().cpu()))
            query_updates.append(query_tensor.detach().clone().cpu())
            sims_steps.append(step + 1)

    updated_sim = F.cosine_similarity(doc_embeddings_tensor, query_tensor.detach()).cpu()

    if save_tensors:  # save query update trajectory
        save_query_updates(
            experiment_dir=experiment_dir,
            topic=topic,
            steps=sims_steps,
            loss_per_step=loss_per_step,
            query_updates=query_updates,
        )
    return updated_sim, doc_to_reranker_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run embedding adaptation experiments with test-time optimization"
    )

    parser.add_argument(
        "--optimize_with_gold",
        type=ast.literal_eval,
        default=False,
        help="Use gold instead of reranker feedback for optimization"
    )
    parser.add_argument(
        "--hyde_model",
        type=str,
        default=None,
        help="LLM model to use for HyDE, when unset HyDE is not used"
    )
    parser.add_argument(
        "--computation_types",
        nargs="+",
        choices=["optimized", "hyde", "optimized_hyde", "all"],
        default=["all"],
        help="Which computation types to run (default: all)"
    )
    # Model parameters
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
        "--embedder_batch_size",
        type=int,
        default=10,
        help="Batch size for embedder inference"
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=10,
        help="Batch size for reranker inference"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="RealScholarQuery",
        choices=Datasets.all_datasets(),
        help="Dataset to use for experiments"
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
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name for output files (default: auto-generated timestamp)"
    )

    parser.add_argument(
        "--save_tensors",
        type=ast.literal_eval,
        default=False,
        help="Whether to save tensors of the query trajectory (default: False)",
    )
    parser.add_argument(
        "--refit",
        type=ast.literal_eval,
        default=False,
        help="Use ReFIT-style optimization: cross-encoder reranker, lr=0.005, temperature=2, min-max normalization",
    )

    args = parser.parse_args()

    if args.refit:
        explicit_args = sys.argv[1:]
        if "--lr" not in explicit_args:
            args.lr = 0.005
        if "--reranker_model" not in explicit_args:
            args.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    return args


if __name__ == '__main__':
    args = parse_args()

    # Get dataset class
    dataset_cls = getattr(Datasets, args.dataset)

    # Extract parameters from args
    embedding_model_name = args.embedding_model
    reranker_model_name = args.reranker_model
    hyde_model_name = args.hyde_model
    lr = args.lr
    num_steps = args.num_steps
    total_scores = args.total_scores
    scores_from_top = args.scores_from_top
    embedder_batch_size = args.embedder_batch_size
    reranker_batch_size = args.reranker_batch_size
    random_seed = args.random_seed
    optimize_with_gold = args.optimize_with_gold
    save_tensors = args.save_tensors

    # Load instruction templates
    templates_yaml_path = os.path.join(os.path.dirname(__file__), "instruction_templates.yaml")
    instruction_templates = load_instruction_templates(templates_yaml_path)
    query_template, document_template = get_templates_for_model(
        embedding_model_name,
        instruction_templates,
        dataset_name=args.dataset
    )
    
    # Get task instruction for LLM reranker
    llm_task_instruction = get_llm_task_instruction(instruction_templates, args.dataset)
    
    print(f"Using query template: {query_template}")
    print(f"Using document template: {document_template}")
    if llm_task_instruction:
        print(f"Using LLM task instruction: {llm_task_instruction}")

    # Setup output directory
    out_dir = os.path.join(os.path.dirname(__file__), "output")

    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.dataset}_{timestamp}"
    else:
        experiment_name = args.experiment_name

    # Create experiment-specific subdirectory
    experiment_dir = os.path.join(out_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create config dict with args and templates
    config_dict = vars(args).copy()
    config_dict['query_template'] = query_template
    config_dict['document_template'] = document_template

    # Save experiment configuration
    save_experiment_config(config_dict, experiment_dir)

    print(f"\n{'='*80}")
    print(f"Starting experiment: {experiment_name}")
    print(f"Output directory: {experiment_dir}")
    print(f"{'='*80}\n")

    # Load input texts, queries and gold relevance labels
    query_to_label_series = dataset_cls().load_gold()
    all_queries = sorted(query_to_label_series.keys())
    print(f"Running on {dataset_cls.dataset_name} dataset, total num topics: {len(all_queries)}, "
          f"total num pairs: {sum(len(s) for s in query_to_label_series.values())}")

    # Determine which computation types to run
    computation_types = args.computation_types
    if "all" in computation_types:
        computation_types = ["optimized", "hyde", "optimized_hyde"]

    print(f"\nComputation types to run: {', '.join(computation_types)}")

    # initialize HyDE model if specified and needed
    hyde_embeddings_list = None
    if "hyde" in computation_types and hyde_model_name is not None:
        print(f"\nInitializing HyDE with model: {hyde_model_name}")
        llm_cache_dir = os.path.join(LLM_CACHE_DIR, "HyDE")
        hyde_llm = get_llm(hyde_model_name, cache_dir=llm_cache_dir)
        print("Generating hypothetical documents for queries...")
        hypothetical_docs = generate_hypothetical_documents(all_queries, hyde_llm)
        print(f"Generated {len(hypothetical_docs)} hypothetical documents")

    if "optimized" in computation_types and not optimize_with_gold:
        # Initialize reranker for generating predictions
        reranker_cache_dir = os.path.join(RERANKER_CACHE_DIR, dataset_cls.dataset_name)
        reranker = get_reranker(reranker_model_name, cache_dir=reranker_cache_dir,
                                task_instruction=llm_task_instruction)
        print(f"Initialized reranker: {reranker_model_name}")
        if llm_task_instruction:
            print(f"  with task instruction: {llm_task_instruction}")
    
    # embed queries
    embedding_model: EmbedderWithFileSystemCache = get_embedder(embedding_model_name,
                                                                cache_dir=os.path.join(CACHE_DIR, dataset_cls.dataset_name))
    device = get_device()
    # Embed original queries
    templated_queries = apply_template(all_queries, query_template)
    topic_embeddings_list = list(zip(all_queries, embedding_model.embed_documents(templated_queries,
                                                                                  batch_size=embedder_batch_size)))

    query_tensors = [torch.tensor(emb, device=device) for _, emb in topic_embeddings_list]

    if "hyde" in computation_types and hyde_model_name is not None:
        # Embed HyDE queries separately (for "hyde" computation type)
        templated_hyde_queries = apply_template(hypothetical_docs, query_template)
        hyde_embeddings_list = list(zip(all_queries, embedding_model.embed_documents(templated_hyde_queries,
                                                                                     batch_size=embedder_batch_size)))
        hyde_query_tensors = [torch.tensor(emb, device=device) for _, emb in hyde_embeddings_list]
    else:
        hyde_query_tensors = [None]*len(query_tensors)

    # Embed all unique documents once before the loop to avoid repeated tensor conversions
    print("\nEmbedding all documents once...")
    all_unique_docs = set()
    for query in all_queries:
        query_gold_series = query_to_label_series[query]
        all_unique_docs.update(query_gold_series.index.tolist())
    all_unique_docs = sorted(all_unique_docs)
    # Apply document template before embedding
    templated_all_docs = apply_template(all_unique_docs, document_template)
    all_doc_embeddings = embedding_model.embed_documents(templated_all_docs, batch_size=embedder_batch_size)

    all_doc_embeddings_tensor = torch.tensor(all_doc_embeddings, device=device)
    doc_to_idx = {doc: idx for idx, doc in enumerate(all_unique_docs)}
    print(f"Document embeddings tensor shape: {all_doc_embeddings_tensor.shape}")

    all_res = defaultdict(list)
    
    # Initialize file writers for streaming raw scores to disk
    raw_scores_paths = {}
    for score_type in ["embeddings", "optimized_embeddings", "hyde", "optimized_hyde_embeddings"]:
        raw_scores_path = os.path.join(experiment_dir, f"{score_type}_raw_scores.parquet")
        if os.path.exists(raw_scores_path):
            os.remove(raw_scores_path)
        raw_scores_paths[score_type] = raw_scores_path
    
    # We'll collect scores in batches and write incrementally
    all_raw_scores = defaultdict(list)
    BATCH_SIZE = 1_000_000  # Write to disk every 1M rows to limit memory usage
    
    # for every query, run test-time optimization and evaluate the results
    for topic, query_tensor, hyde_query_tensor in zip(all_queries, query_tensors, hyde_query_tensors):
        topic_gold_series = query_to_label_series[topic]
        topic_gold_series = topic_gold_series[~topic_gold_series.index.duplicated(keep='first')]
        # Get document embeddings from pre-computed tensor
        doc_indices = [doc_to_idx[doc] for doc in topic_gold_series.index]
        doc_embeddings_tensor = all_doc_embeddings_tensor[doc_indices]

        # Calculate similarities for original embeddings
        similarities = F.cosine_similarity(doc_embeddings_tensor, query_tensor.detach()).cpu()

        orig_embeddings_metrics_dict = evaluate(topic_gold_series.values, similarities.numpy())
        res = (topic, orig_embeddings_metrics_dict)
        print("original", *res)
        all_res["embeddings"].append(res)

        # Evaluate HyDE embeddings separately with HyDE query embeddings
        if "hyde" in computation_types and hyde_embeddings_list is not None:
            # Calculate similarities using HyDE embeddings
            hyde_similarities = F.cosine_similarity(doc_embeddings_tensor, hyde_query_tensor.detach()).cpu()

            hyde_embeddings_metrics_dict = evaluate(topic_gold_series.values, hyde_similarities.numpy())

            res = (topic, hyde_embeddings_metrics_dict)
            print("hyde", *res)
            all_res["hyde"].append(res)
            hyde_score_dict = dict(zip(topic_gold_series.index, hyde_similarities.numpy()))

        # Run optimization if needed
        if "optimized" in computation_types:
            updated_sim, doc_to_reranker_score = optimize(
                doc_embeddings_tensor, topic_gold_series, optimize_with_gold=optimize_with_gold,
                total_scores=total_scores, scores_from_top=scores_from_top, random_seed=random_seed,
                similarities=similarities, query_tensor=query_tensor, lr=lr, num_steps=num_steps,
                reranker=reranker, reranker_batch_size=reranker_batch_size,
                topic=topic, experiment_dir=experiment_dir, save_tensors=save_tensors, refit=args.refit)
            updated_metrics_dict = evaluate(topic_gold_series.values, updated_sim.numpy())
            res = (topic, updated_metrics_dict)
            print("optimized", *res)
            all_res["optimized_embeddings"].append(res)
        else:
            doc_to_reranker_score = {}
            updated_sim = [None]*len(topic_gold_series)

        # optimization over hyde queries
        if "optimized_hyde" in computation_types and hyde_embeddings_list is not None:
            updated_sim_hyde, doc_to_reranker_score_hyde = optimize(
                doc_embeddings_tensor, topic_gold_series, optimize_with_gold=optimize_with_gold,
                total_scores=total_scores, scores_from_top=scores_from_top, random_seed=random_seed,
                similarities=hyde_similarities, query_tensor=hyde_query_tensor, lr=lr, num_steps=num_steps,
                reranker=reranker, reranker_batch_size=reranker_batch_size,
                topic=topic, experiment_dir=experiment_dir, save_tensors=False, refit=args.refit)
            optimized_hyde_metrics_dict = evaluate(topic_gold_series.values, updated_sim_hyde.numpy())
            res = (topic, optimized_hyde_metrics_dict)
            print("optimized_hyde", *res)
            all_res["optimized_hyde_embeddings"].append(res)
            optimized_hyde_score_dict = dict(zip(topic_gold_series.index, updated_sim_hyde.numpy()))
        else:
            doc_to_reranker_score_hyde = {}
            updated_sim_hyde = [None]*len(topic_gold_series)

        # Store raw scores for all documents
        # Also include reranker/gold scores for documents that have them
        for doc, orig_score, updated_score, gold_label \
                in zip(topic_gold_series.index, similarities, updated_sim, topic_gold_series.values):
            base_dict = {
                "topic": topic,
                "document": doc,
                "gold_label": gold_label,
                "reranker_score": doc_to_reranker_score.get(doc, None),
            }
            all_raw_scores["embeddings"].append({**base_dict, "score": orig_score})
            if "optimized" in computation_types:
                all_raw_scores["optimized_embeddings"].append({**base_dict, "score": updated_score})

            # Store HyDE raw scores separately using HyDE similarities
            if "hyde" in computation_types and hyde_embeddings_list is not None:
                hyde_base_dict = base_dict.copy()
                hyde_base_dict["hyde_reranker_score"] = doc_to_reranker_score_hyde.get(doc, None)
                hyde_score = hyde_score_dict[doc]
                all_raw_scores["hyde"].append({**hyde_base_dict, "score": hyde_score})
                if "optimized_hyde" in computation_types:
                    hyde_updated_score = optimized_hyde_score_dict[doc]
                    all_raw_scores["optimized_hyde_embeddings"].append({**hyde_base_dict, "score": hyde_updated_score})

        # Write batches to disk to limit memory usage
        for score_type in list(all_raw_scores.keys()):
            if len(all_raw_scores[score_type]) >= BATCH_SIZE:
                _write_scores_batch(all_raw_scores[score_type], raw_scores_paths[score_type])
                all_raw_scores[score_type] = []  # Clear the batch

    # Save metrics results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")

    for cfg, res_list in all_res.items():
        print(f"\n{cfg} results:")
        all_res_df = pd.DataFrame([{"topic": r[0], **r[1], **vars(args)} for r in res_list])
        print(all_res_df._get_numeric_data().mean())

        out_path = os.path.join(experiment_dir, f"{cfg}_results.csv")
        all_res_df.to_csv(out_path, index=False)
        print(f"Saved {cfg} results to {out_path}")
    
    # Save remaining raw scores for all documents
    for score_type, scores_list in all_raw_scores.items():
        if scores_list:  # Only write if there are remaining scores
            _write_scores_batch(scores_list, raw_scores_paths[score_type])
        print(f"Saved raw {score_type} scores to {raw_scores_paths[score_type]}")
    
    print(f"\n{'='*80}")
    print(f"All results saved to: {experiment_dir}")
    print(f"{'='*80}\n")
