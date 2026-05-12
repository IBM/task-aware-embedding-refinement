from abc import ABC, abstractmethod
import os

from pathlib import Path
from typing import Type

import torch
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from file_system_cache import FileSystemCache, CachedBatchInferenceMixin
from utils import get_device, hash_text

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class BaseEmbedder(ABC):
    """
    Abstract base class for embedding model configurations.
    
    Subclass this to create custom embedder configurations with different
    model parameters, tokenizer settings, or device configurations.
    
    Example:
        class MyCustomEmbedder(BaseEmbedder):
            def create_embedder(self, model_name: str, **kwargs) -> HuggingFaceEmbeddings:
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={"device": self.get_device(), "custom_param": True},
                    **kwargs
                )
    """
        
    @abstractmethod
    def create_embedder(self, model_name: str, **kwargs) -> object:
        """
        Create and return a class instance that implements an `embed_documents` method.
        """
        pass


class DefaultEmbedder(BaseEmbedder):
    def __init__(self, max_input_length: int = 8192):
        self.max_input_length = max_input_length
    
    def create_embedder(self, model_name: str, **kwargs) -> HuggingFaceEmbeddings:
        # Set input max length to avoid OOM issues
        max_length = min(
            self.max_input_length,
            AutoTokenizer.from_pretrained(model_name).model_max_length
        )
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": get_device(),
                "trust_remote_code": True,
                "tokenizer_kwargs": {"model_max_length": max_length},
            },
            **kwargs
        )


class NemotronSentenceTransformersEmbedder(BaseEmbedder):
    """
    Custom embedder for Nemotron models using SentenceTransformer library.
    """

    def create_embedder(self, model_name: str, **kwargs) -> object:
        return NemotronSentenceTransformersWrapper(model_name)


class NemotronSentenceTransformersWrapper:
    """
    Specialized wrapper to properly support llama-embed-nemotron-8b
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

        # SentenceTransformer default is SDPA which is problematic with this model
        attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"

        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"attn_implementation": attn_implementation, "dtype": "bfloat16"},
            tokenizer_kwargs={"padding_side": "left",
                              "model_max_length": 8192},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode_document(texts)
        return embeddings.tolist()


class EmbedderWithFileSystemCache(CachedBatchInferenceMixin[list[float]]):
    """
    Caching wrapper for embedding models with file system persistence.
    
    This class wraps any BaseEmbedder subclass and adds file system caching
    to avoid recomputing embeddings for the same text.
    
    To use a custom embedder configuration, subclass this class and set
    the `embedder_cls` class attribute to your custom BaseEmbedder subclass.
    
    Example:
        class MyCustomCachedEmbedder(EmbedderWithFileSystemCache):
            embedder_cls = MyCustomEmbedder
    """
    embedder_cls: Type[BaseEmbedder] = DefaultEmbedder
    
    def __init__(self, model_name: str, cache_dir: str | Path, **hf_kwargs) -> None:
        """
        Args:
            model_name: The Hugging Face model name (e.g., "sentence-transformers/all-MiniLM-L6-v2").
            cache_dir: Path to a directory to store cache files.
            **hf_kwargs: Any additional kwargs forwarded to the embedder's create_embedder method.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate the embedder class and create the HuggingFaceEmbeddings instance
        embedder_instance = self.embedder_cls()
        self.embedder = embedder_instance.create_embedder(model_name, **hf_kwargs)
        
        model_name_for_cache = model_name.split("/")[1]
        cache_path = self.cache_dir / f"{model_name_for_cache}_embeddings.npz"

        # Set up cache and operation name for the mixin
        self._cache = FileSystemCache[list[float]](cache_path, storage_format="vector")

    def _get_hash_for_input(self, input_text: str) -> str:
        """Generate hash for an input text."""
        return hash_text(input_text)
    
    def _run_inference(self, inputs: list[str]) -> list[list[float]]:
        """Run embedding inference on a batch of inputs."""
        return self.embedder.embed_documents(inputs)

    def embed_documents(self, documents: list[str], batch_size: int = 10) -> list[list[float]]:
        """
        Embed a list of documents with caching.
        
        Args:
            documents: List of document strings to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings (one per document)
        """
        return self._process_batch_with_cache(documents, batch_size)

    def cached_count(self) -> int:
        return len(self._cache)


def get_embedder(model_name, cache_dir):
    # Use NemotronSentenceTransformersEmbedder for llama-embed-nemotron-8b
    if 'llama-embed-nemotron-8b' in model_name:
        class NemotronCachedEmbedder(EmbedderWithFileSystemCache):
            embedder_cls = NemotronSentenceTransformersEmbedder
        return NemotronCachedEmbedder(model_name, cache_dir)
    
    return EmbedderWithFileSystemCache(model_name, cache_dir)
