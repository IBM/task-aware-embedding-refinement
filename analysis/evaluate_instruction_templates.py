"""
Evaluate different instruction templates for embedding models.

This script tests various instruction template formats to find the most effective
way to format queries and documents for a given embedding model and dataset.
"""

import argparse
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd
import torch
import torch.nn.functional as F

from dataset_loaders import Datasets
from embedders import EmbedderWithFileSystemCache, get_embedder, CACHE_DIR
from embedding_adaptation import evaluate, _write_scores_batch
from utils import load_instruction_templates, apply_template, save_experiment_config, get_device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate different instruction templates for embedding models"
    )

    # Model parameters
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name of the embedding model to use"
    )
    parser.add_argument(
        "--embedder_batch_size",
        type=int,
        default=10,
        help="Batch size for embedder inference"
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        default="RealScholarQuery",
        choices=Datasets.all_datasets(),
        help="Dataset to use for experiments"
    )

    # Template configuration
    parser.add_argument(
        "--template_config",
        type=str,
        default="instruction_template_experiments.yaml",
        help="Path to YAML file containing template experiments"
    )
    parser.add_argument(
        "--templates",
        nargs="+",
        default=None,
        help="Specific template names to evaluate (default: all templates in config)"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name for output files (default: auto-generated timestamp)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Get dataset class
    dataset_cls = getattr(Datasets, args.dataset)

    # Extract parameters from args
    embedding_model_name = args.embedding_model
    embedder_batch_size = args.embedder_batch_size

    # Load instruction templates from main config (for dataset-specific instructions)
    main_templates_path = os.path.join(os.path.dirname(__file__), os.path.pardir, "instruction_templates.yaml")
    main_templates = load_instruction_templates(main_templates_path)

    # Load template experiment configurations
    template_config_path = os.path.join(os.path.dirname(__file__), os.path.pardir, args.template_config)
    template_config = load_instruction_templates(template_config_path)
    template_experiments = template_config['experiments']
    
    # Get instruction variants from experiment config, fall back to main templates
    dataset_cfg = main_templates['dataset_instructions'][args.dataset]
    if 'dataset_instruction_variants' in template_config and args.dataset in template_config['dataset_instruction_variants']:
        # Use instruction variants list from experiments YAML
        instruction_variants = template_config['dataset_instruction_variants'][args.dataset]
        if not isinstance(instruction_variants, list):
            raise ValueError(f"Dataset {args.dataset} instruction_variants must be a list in experiments YAML")
    else:
        # Fall back to main templates - wrap single instruction in a list
        instruction_variants = [dataset_cfg.get('instruction', '')]

    print(f"Dataset {args.dataset} has {len(instruction_variants)} instruction variant(s)")
    for i, instr in enumerate(instruction_variants):
        print(f"  Variant {i}: {instr[:80]}..." if len(instr) > 80 else f"  Variant {i}: {instr}")

    # Filter templates if specific ones are requested
    if args.templates:
        template_experiments = {k: v for k, v in template_experiments.items() if k in args.templates}
        if not template_experiments:
            raise ValueError(f"No matching templates found. Available: {list(template_experiments.keys())}")

    print(f"\nEvaluating {len(template_experiments)} template configurations:")
    for name, config in template_experiments.items():
        print(f"  - {name}: {config['description']}")

    # Setup output directory
    out_dir = os.path.join(os.path.dirname(__file__), "output")

    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.dataset}_template_eval_{timestamp}"
    else:
        experiment_name = args.experiment_name

    # Create experiment-specific subdirectory
    experiment_dir = os.path.join(out_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Create config dict with args
    config_dict = vars(args).copy()
    config_dict['template_experiments'] = {k: v for k, v in template_experiments.items()}
    config_dict['instruction_variants'] = instruction_variants

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

    # Initialize embedder
    embedding_model: EmbedderWithFileSystemCache = get_embedder(
        embedding_model_name,
        cache_dir=os.path.join(CACHE_DIR, dataset_cls.dataset_name)
    )
    device = get_device()

    # Get all unique documents
    print("\nCollecting all unique documents...")
    all_unique_docs = set()
    for query in all_queries:
        query_gold_series = query_to_label_series[query]
        all_unique_docs.update(query_gold_series.index.tolist())
    all_unique_docs = sorted(all_unique_docs)
    print(f"Total unique documents: {len(all_unique_docs)}")

    # Dictionary to store results for all template configurations
    all_template_results = {}
    
    # Initialize file writers for streaming raw scores to disk
    # Each combination of (template, instruction_variant) gets its own file
    raw_scores_paths = {}
    for template_name in template_experiments.keys():
        for inst_idx in range(len(instruction_variants)):
            combo_name = f"{template_name}_inst{inst_idx}"
            raw_scores_path = os.path.join(experiment_dir, f"{combo_name}_raw_scores.parquet")
            if os.path.exists(raw_scores_path):
                os.remove(raw_scores_path)
            raw_scores_paths[combo_name] = raw_scores_path
    
    # We'll collect scores in batches and write incrementally
    all_raw_scores = defaultdict(list)
    BATCH_SIZE = 1_000_000  # Write to disk every 1M rows to limit memory usage

    # Outer loop: iterate over instruction variants
    for inst_idx, dataset_instruction in enumerate(instruction_variants):
        print(f"\n{'='*80}")
        print(f"INSTRUCTION VARIANT {inst_idx}/{len(instruction_variants)}")
        print(f"Instruction: {dataset_instruction}")
        print(f"{'='*80}\n")

        # Evaluate each template configuration with this instruction
        for template_name, template_config in template_experiments.items():
            combo_name = f"{template_name}_inst{inst_idx}"
            
            print(f"\n{'-'*80}")
            print(f"Evaluating template: {template_name} (with instruction variant {inst_idx})")
            print(f"Description: {template_config['description']}")
            print(f"Query template: {template_config['query_template']}")
            print(f"Document template: {template_config['document_template']}")
            print(f"{'-'*80}\n")

            # Prepare templates with dataset instruction
            query_template = template_config['query_template']
            document_template = template_config['document_template']

            # Replace {instruction} placeholder if present
            if '{instruction}' in query_template:
                query_template = query_template.replace('{instruction}', dataset_instruction)
            if '{instruction}' in document_template:
                document_template = document_template.replace('{instruction}', dataset_instruction)

            # Embed queries with current template
            templated_queries = apply_template(all_queries, query_template)
            print(f"Example templated query:\n{templated_queries[0]}\n")
            
            topic_embeddings = embedding_model.embed_documents(templated_queries, batch_size=embedder_batch_size)
            topic_embeddings_list = list(zip(all_queries, topic_embeddings))

            # Embed documents with current template
            templated_docs = apply_template(all_unique_docs, document_template)
            print(f"Example templated document:\n{templated_docs[0][:200]}...\n")
            
            all_doc_embeddings = embedding_model.embed_documents(templated_docs, batch_size=embedder_batch_size)

            # Create tensors for efficient computation
            all_doc_embeddings_tensor = torch.tensor(all_doc_embeddings, device=device)
            doc_to_idx = {doc: idx for idx, doc in enumerate(all_unique_docs)}

            print(f"Query embeddings shape: {len(topic_embeddings_list)} x {len(topic_embeddings_list[0][1])}")
            print(f"Document embeddings shape: {all_doc_embeddings_tensor.shape}")

            # Evaluate for each query
            template_results = []
            for topic, query_embedding in topic_embeddings_list:
                topic_gold_series = query_to_label_series[topic]
                topic_gold_series = topic_gold_series[~topic_gold_series.index.duplicated(keep='first')]

                # Get document embeddings from pre-computed tensor
                doc_indices = [doc_to_idx[doc] for doc in topic_gold_series.index]
                doc_embeddings_tensor = all_doc_embeddings_tensor[doc_indices]

                # Calculate similarities
                query_tensor = torch.tensor(query_embedding, device=device)
                similarities = F.cosine_similarity(doc_embeddings_tensor, query_tensor).cpu().numpy()

                # Store raw scores for all documents
                for doc, score, gold_label in zip(topic_gold_series.index, similarities, topic_gold_series.values):
                    all_raw_scores[combo_name].append({
                        "topic": topic,
                        "document": doc,
                        "gold_label": gold_label,
                        "score": float(score),
                    })

                # Evaluate
                metrics_dict = evaluate(topic_gold_series.values, similarities)
                result = (topic, metrics_dict)
                template_results.append(result)
            
            # Write batch to disk to limit memory usage
            if len(all_raw_scores[combo_name]) >= BATCH_SIZE:
                _write_scores_batch(all_raw_scores[combo_name], raw_scores_paths[combo_name])
                all_raw_scores[combo_name] = []  # Clear the batch

            # Store results for this template+instruction combination
            all_template_results[combo_name] = template_results

            # Print summary statistics
            results_df = pd.DataFrame([{"topic": r[0], **r[1]} for r in template_results])
            print(f"\n{combo_name} - Mean metrics:")
            print(results_df._get_numeric_data().mean())
            print()
        print()

    # Save results for all template+instruction combinations
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}\n")

    # Save remaining raw scores for all combinations
    for combo_name, scores_list in all_raw_scores.items():
        if scores_list:  # Only write if there are remaining scores
            _write_scores_batch(scores_list, raw_scores_paths[combo_name])
        print(f"Saved raw {combo_name} scores to {raw_scores_paths[combo_name]}")

    # Create comparison dataframe with all template+instruction combinations
    comparison_data = []
    for combo_name, results_list in all_template_results.items():
        for topic, metrics in results_list:
            row = {
                "template_instruction": combo_name,
                "topic": topic,
                **metrics
            }
            comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save detailed results
    detailed_results_path = os.path.join(experiment_dir, "detailed_results.csv")
    comparison_df.to_csv(detailed_results_path, index=False)
    print(f"Saved detailed results to {detailed_results_path}")

    # Create summary statistics by template+instruction combination
    summary_stats = comparison_df.groupby('template_instruction').agg({
        col: ['mean', 'std'] for col in comparison_df.columns 
        if col not in ['template_instruction', 'topic'] and pd.api.types.is_numeric_dtype(comparison_df[col])
    })

    summary_path = os.path.join(experiment_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_path)
    print(f"Saved summary statistics to {summary_path}")

    # Print summary to console
    print(f"\n{'='*80}")
    print("Summary Statistics (mean across all topics)")
    print(f"{'='*80}\n")
    
    # Get mean metrics for each template+instruction combination
    mean_metrics = comparison_df.groupby('template_instruction').mean(numeric_only=True)
    print(mean_metrics.to_string())
    
    # Find the best combination for each metric
    print(f"\n{'='*80}")
    print("Best Template+Instruction Combination per Metric")
    print(f"{'='*80}\n")
    
    for metric in mean_metrics.columns:
        best_combo = mean_metrics[metric].idxmax()
        best_value = mean_metrics[metric].max()
        print(f"{metric}: {best_combo} ({best_value:.4f})")

    print(f"\n{'='*80}")
    print(f"All results saved to: {experiment_dir}")
    print(f"{'='*80}\n")
