import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

from file_system_cache import FileSystemCache, CachedBatchInferenceMixin
from generative_llms import LLMJudgeVerbalizedScores, LLMJudgeYesNoProbs, LLMRelevanceJudge, INFERENCE_SERVICES
from utils import get_device, hash_text

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class BaseReranker(ABC):
    """
    Abstract base class for reranker model configurations.
    
    Subclass this to create custom reranker configurations with different
    model parameters, tokenizer settings, or device configurations.
    
    Example:
        class MyCustomReranker(BaseReranker):
            def create_reranker(self, model_name: str, **kwargs) -> object:
                return CustomRerankerModel(
                    model_name=model_name,
                    device=get_device(),
                    **kwargs
                )
    """
        
    @abstractmethod
    def create_reranker(self, model_name: str, **kwargs) -> object:
        """
        Create and return a class instance that implements a `get_relevance_scores` method.
        
        The get_relevance_scores method should accept:
            - query: str
            - documents: list[str]
        
        And return:
            - scores: list[float]
        """
        pass

        
class LLMReranker(BaseReranker):
    """
    LLM-based reranker that combines inference services with judge classes.
    
    Uses the INFERENCE_SERVICES registry from generative_llms.py to support
    multiple inference backends (OpenAI, LiteLLM etc.).
    """
    reranking_judge_cls: Type[LLMRelevanceJudge]

    def __init__(self, inference_service="LiteLLM", task_instruction=None):
        self.inference_service = inference_service
        self.task_instruction = task_instruction
        if self.inference_service not in INFERENCE_SERVICES:
            available = ", ".join(INFERENCE_SERVICES.keys())
            raise ValueError(
                f"Unknown inference service: {inference_service}. "
                f"Available services: {available}"
            )
        self.inference_class = INFERENCE_SERVICES[self.inference_service]

    def create_reranker(self, model_name: str, **kwargs) -> LLMRelevanceJudge:
        class CombinedLLMRerankerClass(self.inference_class, self.reranking_judge_cls):
            pass
        
        CombinedLLMRerankerClass.__name__ = self.__class__.__name__
        # Pass task_instruction to the combined class if it was provided
        if self.task_instruction is not None:
            kwargs['task_instruction'] = self.task_instruction
        return CombinedLLMRerankerClass(model_name=model_name, **kwargs)


class LLMYesNoReranker(LLMReranker):
    reranking_judge_cls = LLMJudgeYesNoProbs


class LLMVerbalizedScoreReranker(LLMReranker):
    reranking_judge_cls = LLMJudgeVerbalizedScores


class CrossEncoderReranker(BaseReranker):
    def create_reranker(self, model_name: str, **kwargs) -> object:
        return CrossEncoderRerankerModel(model_name=model_name)


class CrossEncoderRerankerModel:
    def __init__(self, model_name: str, **kwargs):
        self.model = CrossEncoder(model_name, device=get_device())

    def get_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        return self.model.predict(pairs).tolist()


class FollowIRReranker(BaseReranker):
    """
    Reranker implementation for FollowIR-7B model.
    
    FollowIR-7B is a instruction-following reranking model from JHU-CLSP that
    uses a causal language model to predict relevance as true/false tokens.
    
    Reference: https://huggingface.co/jhu-clsp/FollowIR-7B
    """
    
    def create_reranker(self, model_name: str, **kwargs) -> object:
        """
        Create a FollowIR reranker instance.
        
        Args:
            model_name: Should be "jhu-clsp/FollowIR-7B"
            **kwargs: Additional arguments (device, etc.)
        """
        return FollowIRRerankerModel(model_name=model_name, **kwargs)


class FollowIRRerankerModel:
    """
    Implementation of FollowIR-7B reranker model.
    
    This model uses a causal LM to predict relevance by comparing the logits
    of "true" and "false" tokens in response to a relevance query.
    """
    
    def __init__(self, model_name: str = "jhu-clsp/FollowIR-7B", **kwargs):
        """
        Initialize the FollowIR reranker.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on ("cuda" or "cpu")
            **kwargs: Additional arguments passed to model loading
        """
        self.model_name = model_name
        self.device = get_device()
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get token IDs for "true" and "false"
        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id = self.tokenizer.get_vocab()["true"]
        
        # Template from FollowIR paper/repo
        self.template = """<s> [INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.

Query: {query}
Document: {text}
Relevant (only output one word, either "true" or "false"): [/INST] """
    
    def get_relevance_scores(self, query: str, documents: list[str]) -> list[float]:
        """
        Get relevance scores for a list of documents given a query.
        
        Args:
            query: The search query
            documents: List of document texts to score
            
        Returns:
            List of relevance scores between 0 and 1
        """
        # Create prompts for all documents
        prompts = [
            self.template.format(query=query, text=doc)
            for doc in documents
        ]
        
        # Tokenize
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=None,
        )
        
        # Move to device
        tokens = {key: val.to(self.device) for key, val in tokens.items()}
        
        # Get model predictions
        with torch.no_grad():
            logits = self.model(**tokens).logits[:, -1, :]
        
        true_logits = logits[:, self.token_true_id]
        false_logits = logits[:, self.token_false_id]
        
        # Stack and apply softmax to get probabilities
        stacked_logits = torch.stack([false_logits, true_logits], dim=1)
        probs = stacked_logits.softmax(dim=1)
        
        # Get probability of "true" (relevance score)
        scores = probs[:, 1].tolist()
        
        return scores
       

class RerankerWithFileSystemCache(CachedBatchInferenceMixin[float]):
    """
    Caching wrapper for reranker models with file system persistence.
    
    This class wraps any BaseReranker subclass and adds file system caching
    to avoid recomputing reranking scores for the same query-document pairs.
    
    To use a custom reranker configuration, subclass this class and set
    the `reranker_cls` class attribute to your custom BaseReranker subclass.
    
    Example:
        class MyCustomCachedReranker(RerankerWithFileSystemCache):
            reranker_cls = MyCustomReranker
    """
    
    def __init__(self, model_name: str, cache_dir: str | Path,
                 reranker_cls: Type[BaseReranker], task_instruction=None,
                 inference_service=None, **kwargs) -> None:
        """
        Args:
            model_name: The Hugging Face model name (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2").
            cache_dir: Path to a directory to store cache files.
            task_instruction: Optional task instruction to customize the prompt.
            inference_service: Optional inference service name (e.g., "OpenAI", "LiteLLM").
                             Only applicable for LLM-based rerankers.
            **kwargs: Any additional kwargs forwarded to the reranker's create_reranker method.
        """
        self.model_name = model_name
        self.task_instruction = task_instruction
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate the reranker class and create the reranker instance
        # Pass task_instruction and inference_service if the reranker class supports it
        if issubclass(reranker_cls, LLMReranker):
            reranker_kwargs = {'task_instruction': task_instruction}
            if inference_service is not None:
                reranker_kwargs['inference_service'] = inference_service
            reranker_instance = reranker_cls(**reranker_kwargs)
        else:
            reranker_instance = reranker_cls()
        self.reranker = reranker_instance.create_reranker(model_name, **kwargs)
        
        # Include reranker class name in cache path to separate caches for different reranker types
        model_name_for_cache = model_name.split("/")[-1]
        reranker_cls_name = reranker_cls.__name__
        cache_path = self.cache_dir / f"{model_name_for_cache}_{reranker_cls_name}_reranker_scores.npz"

        # Set up cache and operation name for the mixin
        self._cache = FileSystemCache[float](cache_path, storage_format="scalar")
        
        # Store query for use in inference function
        self._current_query: str | None = None

    def _get_hash_for_input(self, input_text: str) -> str:
        """Hash a document with the current query and task instruction for caching."""
        if self._current_query is None:
            raise RuntimeError("Query not set for hashing")

        if self.task_instruction:
            return self._hash_triple(self._current_query, input_text, self.task_instruction)
        else:
            return self._hash_pair(self._current_query, input_text)
    
    def _run_inference(self, inputs: list[str]) -> list[float]:
        """Run reranking inference on a batch of documents."""
        if self._current_query is None:
            raise RuntimeError("Query not set for reranking")
        return self.reranker.get_relevance_scores(self._current_query, inputs)

    def rerank(self, query: str, documents: list[str], batch_size: int = 10) -> list[float]:
        """
        Rerank documents for a given query.
        
        Args:
            query: The query string
            documents: List of document strings to rerank
            batch_size: Batch size for processing
            
        Returns:
            List of relevance scores (one per document)
        """
        # Set the current query for use in inference and hashing
        self._current_query = query
        try:
            return self._process_batch_with_cache(documents, batch_size)
        finally:
            self._current_query = None

    def cached_count(self) -> int:
        return len(self._cache)

    @staticmethod
    def _hash_pair(query: str, document: str) -> str:
        """Hash a query-document pair for caching."""
        combined = f"{query}|||{document}"
        return hash_text(combined)
    
    @staticmethod
    def _hash_triple(query: str, document: str, task_instruction: str) -> str:
        """Hash a query-document-instruction triple for caching."""
        combined = f"{query}|||{document}|||{task_instruction}"
        return hash_text(combined)


def get_reranker(model_name: str, cache_dir: str | Path, task_instruction=None):
    """
    Get a reranker instance with optional task instruction.
    
    Args:
        model_name: The model name to use for reranking.
                   Can include inference service prefix (e.g., "OpenAI/gpt-4").
        cache_dir: Directory for caching reranking scores.
        task_instruction: Optional task instruction to customize the LLM prompt.
                         Only applicable for LLM-based rerankers.
    
    Returns:
        A RerankerWithFileSystemCache instance.
    """
    inference_service = None
    if model_name == "jhu-clsp/FollowIR-7B":
        reranker_cls = FollowIRReranker
    elif model_name.startswith("cross-encoder/"):
        reranker_cls = CrossEncoderReranker
    else:
        reranker_cls = LLMYesNoReranker
        
        # Detect inference service from model name prefix and strip it
        for service_name in INFERENCE_SERVICES.keys():
            if model_name.startswith(f"{service_name}/"):
                inference_service = service_name
                model_name = model_name[len(service_name) + 1:]
                break
    
    return RerankerWithFileSystemCache(
        model_name, cache_dir,
        reranker_cls=reranker_cls,
        task_instruction=task_instruction,
        inference_service=inference_service
    )


if __name__ == '__main__':
    from dataset_loaders import ArgKP21
    import random
    
    # Load ArgKP dataset
    print("Loading ArgKP dataset...")
    kpm = ArgKP21()
    query_to_label_series = kpm.load_gold(splits=("dev",))
    
    # Sample a few queries
    random.seed(42)
    sample_queries = random.sample(list(query_to_label_series.keys()), min(3, len(query_to_label_series)))
    
    # Initialize reranker
    model_name = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
    reranker_cls = LLMYesNoReranker
    reranker = RerankerWithFileSystemCache(reranker_cls=reranker_cls, model_name=model_name,
                                           cache_dir="/Users/arielgera/Documents/Workspaces/query-refinement/cache/tmp")
    
    # Test on real examples
    for query in sample_queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Get documents and labels for this query
        label_series = query_to_label_series[query]
        
        # Sample up to 5 documents (mix of relevant and non-relevant)
        relevant_docs = label_series[label_series == 1].index.tolist()
        non_relevant_docs = label_series[label_series == 0].index.tolist()
        
        sample_docs = []
        sample_labels = []
        
        # Take up to 3 relevant and 2 non-relevant
        for doc in relevant_docs[:3]:
            sample_docs.append(doc)
            sample_labels.append(1)
        for doc in non_relevant_docs[:2]:
            sample_docs.append(doc)
            sample_labels.append(0)
        
        if not sample_docs:
            print("No documents found for this query, skipping...")
            continue
        
        # Get relevance scores from LLM
        print(f"\nEvaluating {len(sample_docs)} documents...")
        scores = reranker.rerank(query, sample_docs)
        if reranker_cls == LLMVerbalizedScoreReranker:
            scores = [s/100 for s in scores]

        # Display results
        for i, (doc, score, label) in enumerate(zip(sample_docs, scores, sample_labels)):
            print(f"\nDocument {i+1} (Gold label: {label}):")
            print(f"  Text: {doc[:100]}..." if len(doc) > 100 else f"  Text: {doc}")
            print(f"  Reranker Score: {score}")
            print(f"  Match: {'✓' if (score > 0.5 and label == 1) or (score <= 0.5 and label == 0) else '✗'}")
