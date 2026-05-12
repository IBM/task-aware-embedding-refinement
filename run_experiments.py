"""
Script to launch multiple runs of embedding_adaptation.py for different models and datasets.
Supports both sequential and parallel execution.
"""

import argparse
import ast
import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime


# Default configurations
DEFAULT_MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-8B",
    "nvidia/llama-embed-nemotron-8b",
    "intfloat/e5-mistral-7b-instruct",
    "Linq-AI-Research/Linq-Embed-Mistral"
]

DEFAULT_DATASETS = [
    "RealScholarQuery",
    "KPM",
    "FollowIR",
    "Clinc150",
    "Banking77",
    "NFCorpus"
]

DEFAULT_RERANKER = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"


def run_experiment(
    embedding_model,
    dataset,
    reranker_model=DEFAULT_RERANKER,
    hyde_model=None,
    computation_types=None,
    lr=1e-4,
    num_steps=100,
    total_scores=20,
    scores_from_top=20,
    embedder_batch_size=10,
    reranker_batch_size=10,
    random_seed=42,
    optimize_with_gold=False,
    save_tensors=False,
    refit=False,
    experiment_name=None,
    dry_run=False,
):
    """
    Run a single experiment with the given configuration.
    
    Args:
        embedding_model: Name of the embedding model
        dataset: Name of the dataset
        reranker_model: Name of the reranker model
        hyde_model: Optional LLM model for HyDE
        computation_types: List of computation types to run
        lr: Learning rate
        num_steps: Number of optimization steps
        total_scores: Total number of documents to sample
        scores_from_top: Number of top documents to include
        embedder_batch_size: Batch size for embedder
        reranker_batch_size: Batch size for reranker
        random_seed: Random seed
        optimize_with_gold: Whether to use gold labels for optimization
        save_tensors: Whether to save query trajectory tensors
        refit: Use ReFIT-style optimization
        experiment_name: Optional custom experiment name
        dry_run: If True, print command without executing
    
    Returns:
        subprocess.CompletedProcess or None if dry_run
    """
    # Build experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = embedding_model.split("/")[-1]
        experiment_name = f"{dataset}_{model_short}_{timestamp}"
    
    # Build command
    cmd = [
        sys.executable,  # Use the same Python interpreter
        "embedding_adaptation.py",
        "--embedding_model", embedding_model,
        "--dataset", dataset,
        "--reranker_model", reranker_model,
        "--lr", str(lr),
        "--num_steps", str(num_steps),
        "--total_scores", str(total_scores),
        "--scores_from_top", str(scores_from_top),
        "--embedder_batch_size", str(embedder_batch_size),
        "--reranker_batch_size", str(reranker_batch_size),
        "--random_seed", str(random_seed),
        "--optimize_with_gold", str(optimize_with_gold),
        "--experiment_name", experiment_name,
    ]
    
    # Add optional arguments
    if hyde_model:
        cmd.extend(["--hyde_model", hyde_model])
    
    if computation_types:
        cmd.extend(["--computation_types"] + computation_types)

    if save_tensors:
        cmd.extend(["--save_tensors", str(save_tensors)])

    if refit:
        cmd.extend(["--refit", str(refit)])

    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_name}")
    print(f"  Model: {embedding_model}")
    print(f"  Dataset: {dataset}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("DRY RUN - Command not executed")
        return None
    
    # Run the command
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ Experiment completed successfully: {experiment_name}\n")
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment failed: {experiment_name}")
        print(f"  Error code: {e.returncode}\n")
        raise


def run_single_experiment(exp_config, args):
    """Wrapper function for running experiments in parallel."""
    i, model, dataset = exp_config

    print(f"\n{'#' * 80}")
    print(f"# Experiment {i}")#/{len(combinations)}")
    print(f"{'#' * 80}")

    # Generate experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Add microseconds for uniqueness
    model_short = model.split("/")[-1]
    experiment_name = f"{dataset}_{model_short}_{timestamp}"
    if args.experiment_prefix:
        experiment_name = f"{args.experiment_prefix}_{experiment_name}"

    try:
        run_experiment(
            embedding_model=model,
            dataset=dataset,
            reranker_model=args.reranker_model,
            hyde_model=args.hyde_model,
            computation_types=args.computation_types,
            lr=args.lr,
            num_steps=args.num_steps,
            total_scores=args.total_scores,
            scores_from_top=args.scores_from_top,
            embedder_batch_size=args.embedder_batch_size,
            reranker_batch_size=args.reranker_batch_size,
            random_seed=args.random_seed,
            optimize_with_gold=args.optimize_with_gold,
            save_tensors=args.save_tensors,
            refit=args.refit,
            experiment_name=experiment_name,
            dry_run=args.dry_run,
        )
        return ("success", model, dataset, experiment_name)
    except Exception as e:
        return ("failed", model, dataset, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple runs of embedding_adaptation.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default models on all default datasets
  python run_experiments.py
  
  # Run specific models on specific datasets
  python run_experiments.py --embedding_models "Qwen/Qwen3-Embedding-0.6B" --datasets "Clinc150" "NFCorpus"
  
  # Run 3 experiments in parallel
  python run_experiments.py --parallel 3
  
  # Dry run to see what would be executed
  python run_experiments.py --dry_run
  
  # Run with HyDE
  python run_experiments.py --hyde_model "meta-llama/Llama-3.1-8B-Instruct"
  
  # Run 4 experiments in parallel and continue on errors
  python run_experiments.py --parallel 4 --continue_on_error
        """
    )
    
    # Model and dataset selection
    parser.add_argument(
        "--embedding_models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of embedding models to test"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="List of datasets to test"
    )
    parser.add_argument(
        "--reranker_model",
        type=str,
        default=DEFAULT_RERANKER,
        help="Reranker model to use"
    )
    parser.add_argument(
        "--hyde_model",
        type=str,
        default=None,
        help="LLM model for HyDE (optional)"
    )
    parser.add_argument(
        "--computation_types",
        nargs="+",
        choices=["optimized", "hyde", "all"],
        default=None,
        help="Which computation types to run"
    )
    
    # Optimization parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of optimization steps"
    )
    parser.add_argument(
        "--total_scores",
        type=int,
        default=20,
        help="Total number of documents to sample"
    )
    parser.add_argument(
        "--scores_from_top",
        type=int,
        default=20,
        help="Number of top documents to include"
    )
    
    # Batch sizes
    parser.add_argument(
        "--embedder_batch_size",
        type=int,
        default=10,
        help="Batch size for embedder"
    )
    parser.add_argument(
        "--reranker_batch_size",
        type=int,
        default=10,
        help="Batch size for reranker"
    )
    
    # Other parameters
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--optimize_with_gold",
        action="store_true",
        help="Use gold labels instead of reranker"
    )

    parser.add_argument(
        "--save_tensors",
        type=ast.literal_eval,
        help="Whether to save query trajectory tensors"
    )
    parser.add_argument(
        "--refit",
        type=ast.literal_eval,
        default=False,
        help="Use ReFIT-style optimization: cross-encoder reranker, lr=0.005, temperature=2, min-max normalization"
    )
    
    # Execution control
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running experiments even if one fails"
    )
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default=None,
        help="Prefix for experiment names"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Number of experiments to run in parallel (default: 1 for sequential)"
    )
    
    args = parser.parse_args()
    
    # Generate all combinations of models and datasets
    combinations = list(itertools.product(args.embedding_models, args.datasets))
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT BATCH CONFIGURATION")
    print(f"{'='*80}")
    print(f"Total experiments to run: {len(combinations)}")
    print(f"Models: {', '.join(args.embedding_models)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Reranker: {args.reranker_model}")
    if args.hyde_model:
        print(f"HyDE model: {args.hyde_model}")
    if args.computation_types:
        print(f"Computation types: {', '.join(args.computation_types)}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    # Track results
    successful = []
    failed = []
    
    # Run experiments either sequentially or in parallel
    if args.parallel == 1:
        # Sequential execution
        for i, (model, dataset) in enumerate(combinations, 1):
            result = run_single_experiment((i, model, dataset), args)
            if result[0] == "success":
                successful.append((result[1], result[2], result[3]))
            else:
                failed.append((result[1], result[2], result[3]))
                if not args.continue_on_error:
                    print(f"\n✗ Stopping due to error. Use --continue_on_error to continue on failures.")
                    break
                else:
                    print(f"\n✗ Error occurred but continuing with next experiment...")
    else:
        # Parallel execution
        print(f"Running {args.parallel} experiments in parallel...")
        exp_configs = [(i, model, dataset) for i, (model, dataset) in enumerate(combinations, 1)]
        
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(run_single_experiment, config, args): config
                for config in exp_configs
            }
            
            # Process results as they complete
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result[0] == "success":
                        successful.append((result[1], result[2], result[3]))
                        print(f"\n✓ Completed: {result[3]}")
                    else:
                        failed.append((result[1], result[2], result[3]))
                        print(f"\n✗ Failed: {result[1]} on {result[2]}")
                        if not args.continue_on_error:
                            print(f"\n✗ Stopping due to error. Use --continue_on_error to continue on failures.")
                            # Cancel remaining futures
                            for f in future_to_config:
                                f.cancel()
                            break
                except Exception as e:
                    i, model, dataset = config
                    failed.append((model, dataset, str(e)))
                    print(f"\n✗ Exception in experiment {i}: {e}")
                    if not args.continue_on_error:
                        for f in future_to_config:
                            f.cancel()
                        break
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT BATCH SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {len(combinations)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n✓ Successful experiments:")
        for model, dataset, exp_name in successful:
            print(f"  - {exp_name} ({model} on {dataset})")
    
    if failed:
        print(f"\n✗ Failed experiments:")
        for model, dataset, error in failed:
            print(f"  - {model} on {dataset}: {error}")
    
    print(f"{'='*80}\n")
    
    # Exit with error code if any failed
    if failed and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
