import json
import os
import re
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Union, Literal

import numpy as np
from openai import OpenAI, InternalServerError
from openai.types.chat import ChatCompletion
from tqdm.auto import tqdm
from dotenv import load_dotenv

from file_system_cache import FileSystemCache, CachedBatchInferenceMixin
from utils import run_with_imap, hash_text

load_dotenv()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

ERROR_SCORE = "ERR"

MAX_INPUT_CHARS = 500_000


class RelevanceJudge:
    def get_relevance_scores(self, query: str, documents: list[str]):
        raise NotImplementedError()

    @staticmethod
    def judge_description(model_name):
        return model_name.split('/')[-1]


class LLMRelevanceJudge(RelevanceJudge):
    choices: list[str]
    prompt: str

    def infer(self, inputs, **gen_params):
        raise NotImplementedError()

    @staticmethod
    def get_generated_text(pred_obj) -> Union[str, Literal["ERR"]]:
        raise NotImplementedError()

    @staticmethod
    def get_generated_token_dicts(pred_obj) -> Union[list[dict], Literal["ERR"]]:
        raise NotImplementedError()

    def apply_prompt_template(self, query: str, documents: list[str]):
        inputs = []
        for doc in documents:
            if (len(doc) + len(self.prompt) + len(query)) > MAX_INPUT_CHARS:
                print(f"truncating doc (original length: {len(doc)} characters)")
                doc = doc[:MAX_INPUT_CHARS-len(self.prompt)-len(query)]

            inputs.append(self.prompt.format(query, doc))
        return inputs


class OpenAIStyleMixin(LLMRelevanceJudge):
    def __init__(self, model_name, endpoint_url=None, api_key="EMPTY", random_seed=42,
                 num_parallel_requests=30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_name = model_name.split("#")[0]
        self.random_seed = random_seed
        self.num_parallel_requests = num_parallel_requests

        if endpoint_url and not endpoint_url.endswith("/v1"):
            endpoint_url = f"{endpoint_url}/v1"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint_url,
        )
        # print(self.client.models.list())

    def infer(self, inputs, **gen_params):
        processed_inputs = [[
            {"role": "user", "content": user_input},
        ] for user_input in inputs]

        if self.client.base_url.host == "api.openai.com":
            gen_params["max_completion_tokens"] = gen_params.pop("max_new_tokens", None)
        else:
            gen_params['max_tokens'] = gen_params.pop("max_new_tokens", None)

        if 'return_options' in gen_params:
            gen_params['top_logprobs'] = gen_params['return_options'].get("top_n_tokens")
            gen_params['logprobs'] = gen_params['return_options'].get("token_logprobs") is True
            gen_params.pop("return_options")
                
        predictions = []
        inputs = [(inp, gen_params) for inp in processed_inputs]
        with ThreadPool(processes=self.num_parallel_requests) as pool:
            for pred in tqdm(pool.imap(self.get_completion, inputs), 
                             total=len(inputs), desc=f"Inferring with {self.__class__.__name__}"):
                predictions.append(pred)
        return predictions

    @run_with_imap
    def get_completion(self, processed_input, gen_params):
        if "Qwen" in self.model_name:
            gen_params["extra_body"] = {**gen_params.get("extra_body", {}),
                                        "chat_template_kwargs": {"enable_thinking": False}}
        try:
            completion = self.client.chat.completions.create(
                messages=processed_input,
                model=self.model_name,
                seed=gen_params.pop("seed", self.random_seed),
                **gen_params
            )
            return completion
        except Exception as exception:
            if isinstance(exception, InternalServerError):
                raise exception

            print(exception)
            # Mark content filter errors so they can be handled differently
            is_content_filter = "The response was filtered" in str(exception)
            return dict(error=exception, is_content_filter=is_content_filter)

    @staticmethod
    def get_generated_text(pred_obj: ChatCompletion) -> Union[str, Literal["ERR"]]:
        if not isinstance(pred_obj, ChatCompletion):
            return ERROR_SCORE
        content = pred_obj.choices[0].message.content
        if content is None:
            return ERROR_SCORE
        if content.startswith("{"):
            try:
                score: str = json.loads(content)['score']
            except:
                print("ERROR parsing json", content)
                return ERROR_SCORE
        else:
            score = content
        return score

    @staticmethod
    def get_generated_token_dicts(pred_obj: ChatCompletion):
        if not isinstance(pred_obj, ChatCompletion):
            if isinstance(pred_obj, dict) and pred_obj.get("is_content_filter"):
                return "CONTENT_FILTER_ERROR"
            return ERROR_SCORE
        top_logprobs_response = pred_obj.choices[0].logprobs.content
        token_dicts = [
            {
                "top_tokens": [
                    {"text": obj.token, "logprob": obj.logprob}
                    for obj in generated_token.top_logprobs
                ]
            }
            for generated_token in top_logprobs_response
        ]

        return token_dicts


class OpenAIMixin(OpenAIStyleMixin):
    def __init__(self, model_name, api_key=os.environ.get("OPENAI_API_KEY"), *args, **kwargs):
        super().__init__(model_name, api_key=api_key, *args, **kwargs)


class LiteLLMMixin(OpenAIStyleMixin):
    def __init__(self, model_name, api_key=os.environ.get("API_KEY"), *args, **kwargs):
        endpoint_url = os.environ["BASE_URL"]
        super().__init__(model_name, api_key=api_key, endpoint_url=endpoint_url, *args, **kwargs)


INFERENCE_SERVICES = {
    "OpenAI": OpenAIMixin,
    "LiteLLM": LiteLLMMixin,
}


def create_llm_instance(model_name: str, **kwargs):
    """Create the underlying LLM instance based on model name."""
    for service_name, service_class in INFERENCE_SERVICES.items():
        if model_name.startswith(f"{service_name}/"):
            model_name = model_name[len(service_name) + 1:]
            return service_class(model_name, **kwargs)

    # default to OpenAI
    return LiteLLMMixin(model_name, **kwargs)


class LLMWithFileSystemCache(CachedBatchInferenceMixin[str]):
    """
    Caching wrapper for LLM models with file system persistence.
    
    This class wraps any LLM inference service and adds file system caching
    to avoid recomputing LLM responses for the same inputs.
    """
    
    def __init__(self, model_name: str, cache_dir: str | Path, **kwargs) -> None:
        """
        Args:
            model_name: The model name (e.g., "gpt-4").
            cache_dir: Path to a directory to store cache files.
            **kwargs: Any additional kwargs forwarded to the LLM's constructor.
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create the underlying LLM instance
        self.llm = create_llm_instance(model_name, **kwargs)
        
        # Set up cache path
        model_name_for_cache = model_name.replace("/", "_")
        cache_path = self.cache_dir / f"{model_name_for_cache}_llm_cache.npz"

        # Set up cache with string storage format
        self._cache = FileSystemCache[str](cache_path, storage_format="string")
        
        # Store generation params for use in inference
        self._current_gen_params: dict | None = None

    def _get_hash_for_input(self, input_text: str) -> str:
        """Hash an input with the current generation params for caching."""
        if self._current_gen_params is None:
            raise RuntimeError("Generation params not set for hashing")
        # Include generation params in hash to differentiate different generation settings
        params_str = json.dumps(self._current_gen_params, sort_keys=True)
        combined = f"{input_text}|||{params_str}"
        return hash_text(combined)
    
    def _run_inference(self, inputs: list[str]) -> list[str]:
        """Run LLM inference on a batch of inputs."""
        if self._current_gen_params is None:
            raise RuntimeError("Generation params not set for inference")
        
        # Call the underlying LLM's infer method
        predictions = self.llm.infer(inputs, **self._current_gen_params)
        
        # Extract generated text from predictions
        results = [self.llm.get_generated_text(pred) for pred in predictions]
        
        return results

    def infer(self, inputs: list[str], batch_size: int = 10, **gen_params):
        """
        Run inference on inputs with caching.
        
        Args:
            inputs: List of input strings
            batch_size: Batch size for processing
            **gen_params: Generation parameters (max_new_tokens, temperature, etc.)
            
        Returns:
            List of generated text strings
        """
        # Set the current generation params for use in inference and hashing
        self._current_gen_params = gen_params
        try:
            return self._process_batch_with_cache(inputs, batch_size)
        finally:
            self._current_gen_params = None

    def cached_count(self) -> int:
        return len(self._cache)


def get_llm(model_name: str, cache_dir: str | Path | None = None, **kwargs):
    """
    Get an LLM instance based on model name pattern, optionally with caching.
    
    Args:
        model_name: Model name (e.g., "OpenAI/gpt-4")
        cache_dir: Optional cache directory. If provided, returns cached LLM instance.
        **kwargs: Additional arguments forwarded to the LLM constructor.
    
    Returns an appropriate inference service class instance based on model naming:
    - Models starting with service prefix (e.g., "OpenAI/") use that service
    - Otherwise defaults to LiteLLM
    """
    if cache_dir is not None:
        return LLMWithFileSystemCache(model_name, cache_dir, **kwargs)
    else:
        return create_llm_instance(model_name, **kwargs)


class LLMJudgeYesNoProbs(LLMRelevanceJudge):
    def __init__(self, *args, task_instruction=None, **kwargs):
        task_instruction_suffix = f"Judge the relevance according to the following task: {task_instruction} " if task_instruction else ""
        self.prompt = (
            f"Here is a user query and a document. Is this document relevant for this query? {task_instruction_suffix}"
            f"Answer with only yes/no, without any preceding tokens. Query: {{}}\nDocument: {{}}\nRelevant? (Yes/No):")
        self.num_tokens_to_check = 2
        self.min_probability_mass = 0.0001
        self.choices = ["Yes", "No"]

    def calc_score(self, token_preds: list[dict]):
        # Handle content filter errors with a neutral mock score
        if token_preds == "CONTENT_FILTER_ERROR":
            print("Content filter triggered - returning neutral score 0.5")
            return 0.5
        
        for i in range(min(self.num_tokens_to_check, len(token_preds))):
            try:
                pos_probs, neg_probs = self.get_pos_neg_probs(token_logprobs_obj=token_preds[i]["top_tokens"])
                if pos_probs or neg_probs:
                    sum_probs = sum(pos_probs) + sum(neg_probs)
                    if sum_probs > self.min_probability_mass:
                        return round(float(sum(pos_probs) / sum_probs), 5)
            except:
                pass
        return ERROR_SCORE

    @staticmethod
    def get_pos_neg_probs(token_logprobs_obj):
        pos_and_neg_probs = []
        for class_name in ["yes", "no"]:
            # We need to capture different variants of model behavior and tokenizers, for example with opening space,
            # punctuation etc. but avoid longer words that contain the class name.
            # For example, for class "yes" we would capture "YES," and " Yes" but not "yesterday".
            name_regex = re.compile(
                rf"(\W|Ġ|_)*{class_name}(\W|Ġ|_)*", flags=re.IGNORECASE
            )
            class_probs = [
                np.exp(d["logprob"])
                for d in token_logprobs_obj
                if name_regex.fullmatch(d["text"])
            ]
            pos_and_neg_probs.append(class_probs)
        return pos_and_neg_probs

    def get_relevance_scores(self, query, documents):
        inputs = self.apply_prompt_template(query, documents)
        preds = self.infer(inputs, max_new_tokens=5,
                           return_options={"generated_tokens": True, "token_logprobs": True, "top_n_tokens": 5})
        scores = []
        for pred in preds:
            score = self.calc_score(self.get_generated_token_dicts(pred))
            scores.append(score)
        return scores

    @staticmethod
    def judge_description(model_name):
        short_model_name = model_name.split('/')[-1]
        return f"{short_model_name}_logprob-score"


class LLMJudgeVerbalizedScores(LLMRelevanceJudge):
    def __init__(self, *args, min_score=0, max_score=100, task_instruction=None, **kwargs):
        self.min_score = min_score
        self.max_score = max_score

        task_instruction_suffix = f"Judge the relevance according to the following task: {task_instruction} " if task_instruction else ""
        self.prompt = (
            f"Here is a user query and a document. On a scale of {min_score} to {max_score}, "
            f"to what extent is this document relevant for this query? {task_instruction_suffix}"
            f"Reply with your rating score without any preceding explanation. "
            f"Query: {{}}\nDocument: {{}}\nRelevance rating ({min_score}-{max_score}):")

        self.choices = [str(num) for num in range(min_score, max_score + 1)]

    def get_relevance_scores(self, query, documents, **generation_kwargs):
        inputs = self.apply_prompt_template(query, documents)
        preds = self.infer(inputs, max_new_tokens=5, **generation_kwargs)
        scores = []
        for pred in preds:
            generated_text = self.get_generated_text(pred)
            try:
                match = re.search(r"([-]*[0-9]+(\.([0-9]+))*)|([\w]+)", generated_text)
                score = float(generated_text[match.start():match.end()])
            except:
                print("ERROR parsing response", generated_text)
                score = ERROR_SCORE
            scores.append(score)
        return scores

    @staticmethod
    def judge_description(model_name):
        short_model_name = model_name.split('/')[-1]
        return f"{short_model_name}_verbalized-score"


def generate_hypothetical_documents(queries, llm_model):
    hyde_prompt_template = (
        "Write a representative document passage that would be relevant to the following query. "
        "The passage should capture key concepts and terminology that relevant documents would contain.\n\n"
        "Query: {}\n\n"
        "Passage:"
    )
    
    inputs = [hyde_prompt_template.format(query) for query in queries]
    hypothetical_docs = llm_model.infer(inputs, max_new_tokens=256, temperature=0.7)
    assert len(hypothetical_docs) == len(queries) and ERROR_SCORE not in hypothetical_docs, \
        "Errors in generating HyDE docs"
    return hypothetical_docs
