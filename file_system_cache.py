import tempfile
import os

from collections import defaultdict
from pathlib import Path
from typing import TypeVar, Generic, Callable

import numpy as np
from tqdm import tqdm


T = TypeVar('T')


class FileSystemCache(Generic[T]):
    """
    Generic file system cache for storing computed results.
    
    Supports three storage formats:
    - Vector format: stores dict[str, list[float]] (for embeddings)
    - Scalar format: stores dict[str, float] (for reranker scores)
    - String format: stores dict[str, str] (for LLM responses)
    """
    
    def __init__(self, cache_path: Path, storage_format: str = "vector"):
        """
        Args:
            cache_path: Path to the cache file
            storage_format: Either "vector" (for embeddings), "scalar" (for scores), or "string" (for text)
        """
        self.cache_path = cache_path
        self.storage_format = storage_format
        self._cache = self._load_cache()
    
    def get(self, key: str) -> T | None:
        """Get value from cache by key."""
        return self._cache.get(key)
    
    def set(self, key: str, value: T) -> None:
        """Set value in cache and persist to disk."""
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self._cache)
    
    def _load_cache(self) -> dict[str, T]:
        """Load cache from disk."""
        if not self.cache_path.exists():
            return {}
        
        if self.storage_format == "vector":
            return self._load_vector_cache()
        elif self.storage_format == "scalar":
            return self._load_scalar_cache()
        elif self.storage_format == "string":
            return self._load_string_cache()
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")
    
    def _load_vector_cache(self) -> dict[str, list[float]]:
        """Load vector cache (embeddings)."""
        with np.load(str(self.cache_path), allow_pickle=False) as data:
            keys = data["keys"]
            embs = data["embeddings"]

            if keys.ndim != 1 or embs.ndim != 2 or len(keys) != embs.shape[0]:
                raise ValueError("Invalid vector cache file format")

            # Convert to Python types: dict[str, list[float]]
            cache = {}
            for i, k in enumerate(keys.tolist()):
                cache[str(k)] = embs[i].astype(float).tolist()
            return cache
    
    def _load_scalar_cache(self) -> dict[str, float]:
        """Load scalar cache (scores)."""
        with np.load(str(self.cache_path), allow_pickle=False) as data:
            keys = data["keys"]
            scores = data["scores"]

            if keys.ndim != 1 or scores.ndim != 1 or len(keys) != len(scores):
                raise ValueError("Invalid scalar cache file format")

            # Convert to Python types: dict[str, float]
            cache = {}
            for i, k in enumerate(keys.tolist()):
                cache[str(k)] = float(scores[i])
            return cache
    
    def _load_string_cache(self) -> dict[str, str]:
        """Load string cache (text responses)."""
        with np.load(str(self.cache_path), allow_pickle=False) as data:
            keys = data["keys"]
            strings = data["strings"]

            if keys.ndim != 1 or strings.ndim != 1 or len(keys) != len(strings):
                raise ValueError("Invalid string cache file format")

            # Convert to Python types: dict[str, str]
            cache = {}
            for i, k in enumerate(keys.tolist()):
                cache[str(k)] = str(strings[i])
            return cache
    
    def _save_cache(self) -> None:
        """Save cache to disk atomically."""
        if self.storage_format == "vector":
            self._save_vector_cache()
        elif self.storage_format == "scalar":
            self._save_scalar_cache()
        elif self.storage_format == "string":
            self._save_string_cache()
        else:
            raise ValueError(f"Unknown storage format: {self.storage_format}")
    
    def _save_vector_cache(self) -> None:
        """Save vector cache (embeddings) atomically."""
        keys = list(self._cache.keys())

        # Build a consistent 2D float32 matrix
        first_vec = np.asarray(self._cache[keys[0]], dtype=np.float32)
        dim = int(first_vec.shape[0])
        emb_matrix = np.zeros((len(keys), dim), dtype=np.float32)

        for i, k in enumerate(keys):
            vec = np.asarray(self._cache[k], dtype=np.float32)
            if vec.shape != (dim,):
                raise ValueError(f"Inconsistent embedding dimension for key {k}: expected {dim}, got {vec.shape}")
            emb_matrix[i] = vec

        # Create a fixed-width Unicode dtype to avoid object arrays
        max_len = max(1, max(len(k) for k in keys))
        keys_arr = np.array(keys, dtype=f"<U{max_len}")

        # Atomic write
        self._atomic_write(lambda f: np.savez_compressed(f, keys=keys_arr, embeddings=emb_matrix))
    
    def _save_scalar_cache(self) -> None:
        """Save scalar cache (scores) atomically."""
        keys = list(self._cache.keys())
        scores = np.array([self._cache[k] for k in keys], dtype=np.float32)

        # Create a fixed-width Unicode dtype to avoid object arrays
        max_len = max(1, max(len(k) for k in keys))
        keys_arr = np.array(keys, dtype=f"<U{max_len}")

        # Atomic write
        self._atomic_write(lambda f: np.savez_compressed(f, keys=keys_arr, scores=scores))
    
    def _save_string_cache(self) -> None:
        """Save string cache (text responses) atomically."""
        keys = list(self._cache.keys())
        strings = [self._cache[k] for k in keys]

        # Create fixed-width Unicode dtypes to avoid object arrays
        max_key_len = max(1, max(len(k) for k in keys))
        max_str_len = max(1, max(len(s) for s in strings))
        
        keys_arr = np.array(keys, dtype=f"<U{max_key_len}")
        strings_arr = np.array(strings, dtype=f"<U{max_str_len}")

        # Atomic write
        self._atomic_write(lambda f: np.savez_compressed(f, keys=keys_arr, strings=strings_arr))
    
    def _atomic_write(self, write_func: Callable) -> None:
        """Perform atomic file write using temp file and rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.cache_path.parent),
            prefix=self.cache_path.name,
            suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                write_func(tmp_file)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_path, self.cache_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class CachedBatchInferenceMixin(Generic[T]):
    """
    Mixin class for cached batch inference with deduplication.
    
    This mixin provides caching functionality to classes that implement batch inference.
    Classes using this mixin must:
    1. Set `_cache` attribute (FileSystemCache[T])
    2. Implement `_get_hash_for_input(input: str) -> str` method
    3. Implement `_run_inference(inputs: list[str]) -> list[T]` method
    
    Type Parameters:
        T: The output type for each input item (e.g., list[float] for embeddings)
    """
    
    # These attributes must be set by the class using this mixin
    _cache: FileSystemCache[T]
    _save_every_n_batches: int = 100
    
    def _get_hash_for_input(self, input_text: str) -> str:
        raise NotImplementedError
    
    def _run_inference(self, inputs: list[str]) -> list[T]:
        """
        Run the actual inference on a batch of inputs.
        
        Args:
            inputs: List of input strings to process
            
        Returns:
            List of results (one per input, in same order)
        """
        raise NotImplementedError
    
    def _process_batch_with_cache(self, inputs: list[str], batch_size: int = 10) -> list[T]:
        if len(inputs) <= batch_size:
            return self._process_single_batch(inputs, save_cache=True)

        initial_cache_size = len(self._cache)
        results = []
        for batch_start in tqdm(range(0, len(inputs), batch_size)):
            batch = inputs[batch_start:batch_start + batch_size]
            save_now = batch_start > 0 and ((batch_start % (batch_size*self._save_every_n_batches)) == 0)
            batch_res = self._process_single_batch(batch, save_cache=save_now)
            results.extend(batch_res)
        # always save at end (assuming the cache was updated)
        if len(self._cache) > initial_cache_size:
            self._cache._save_cache()
        return results
    
    def _process_single_batch(self, inputs: list[str], save_cache: bool = True) -> list[T]:
        # Map each input to its hash and track indices for duplicates
        input_hashes = [self._get_hash_for_input(inp) for inp in inputs]
        positions_by_hash = defaultdict(list)
        for idx, h in enumerate(input_hashes):
            positions_by_hash[h].append(idx)
        
        # Determine which hashes are missing from cache
        missing_hashes: list[str] = [
            h for h in positions_by_hash.keys() if h not in self._cache
        ]
        # Pick one representative input for each missing hash
        missing_inputs: list[str] = [
            inputs[positions_by_hash[h][0]] for h in missing_hashes
        ]
        
        # Compute results for missing inputs (if any)
        if missing_inputs:
            new_results: list[T] = self._run_inference(missing_inputs)
            if len(new_results) != len(missing_inputs):
                raise RuntimeError(
                    f"Inference returned a different number of results "
                    f"(expected {len(missing_inputs)}, got {len(new_results)})."
                )
            
            # Update cache in-memory
            new_cache_items = {h: res for h, res in zip(missing_hashes, new_results)}
            self._cache._cache.update(new_cache_items)
            # Save to file system only if requested
            if save_cache:
                self._cache._save_cache()
        
        # Prepare output aligned with the original input order
        result: list[T] = [None] * len(inputs)  # type: ignore
        for h, positions in positions_by_hash.items():
            cached_result = self._cache.get(h)
            if cached_result is None:
                raise RuntimeError(f"Expected result for hash {h} to be in cache")
            for pos in positions:
                result[pos] = cached_result
        
        return result
