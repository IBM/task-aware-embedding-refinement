import argparse
import os

from dataset_loaders import Datasets
from embedders import get_embedder, CACHE_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed all documents for a dataset and embedder"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=Datasets.all_datasets(),
        help="Dataset to embed documents for"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Name of the embedding model to use"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get dataset class
    dataset_cls = getattr(Datasets, args.dataset)
    print(f"\n{'='*80}")
    print(f"Embedding all documents for dataset: {dataset_cls.dataset_name}")
    print(f"Using embedding model: {args.embedding_model}")
    print(f"{'='*80}\n")
    
    
    query_to_label_series = dataset_cls().load_gold()
    all_documents = set()
    for query, label_series in query_to_label_series.items():
        all_documents.update(label_series.index.tolist())
    
    all_documents = sorted(list(all_documents))
    print(f"Total unique documents: {len(all_documents)}")
    
    cache_dir = os.path.join(CACHE_DIR, dataset_cls.dataset_name)
    embedding_model = get_embedder(args.embedding_model, cache_dir=cache_dir)
    
    initial_cache_count = embedding_model.cached_count()
    print(f"Already cached embeddings: {initial_cache_count}")
    
    print(f"\nEmbedding documents (batch_size={args.batch_size})...")
    doc_embeddings = embedding_model.embed_documents(
        all_documents,
        batch_size=args.batch_size
    )
    
    final_cache_count = embedding_model.cached_count()
    newly_embedded = final_cache_count - initial_cache_count
    
    print(f"\n{'='*80}")
    print("Embedding complete!")
    print(f"Total embeddings in cache: {final_cache_count}")
    print(f"Newly embedded documents: {newly_embedded}")
    print(f"Cache directory: {cache_dir}")
    print(f"{'='*80}\n")
    
    # Print some statistics
    print("Document statistics:")
    doc_lengths = [len(doc) for doc in all_documents]
    print(f"  Average document length: {sum(doc_lengths) / len(doc_lengths):.1f} characters")
    print(f"  Min document length: {min(doc_lengths)} characters")
    print(f"  Max document length: {max(doc_lengths)} characters")
    
    # Print embedding statistics
    if doc_embeddings:
        embedding_dim = len(doc_embeddings[0])
        print(f"\nEmbedding dimension: {embedding_dim}")


if __name__ == "__main__":
    main()