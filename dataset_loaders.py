import json
import os
import random
import shutil
import zipfile
import urllib

from collections import defaultdict
from dataclasses import dataclass

from datasets import load_dataset
import pandas as pd
from huggingface_hub import snapshot_download
from tqdm import tqdm


DATA_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_OUT_DIR, exist_ok=True)


class Dataset:
    dataset_name: str

    def load_gold(self):
        raise NotImplementedError


class ArgKP21(Dataset):
    dataset_name = "keypoint_matching"

    def load_gold(self, splits=("dev", "train")):
        """
        Load ArgKP21 (Key Point Matching) dataset from GitHub.
        
        Downloads the dataset from https://github.com/IBM/KPA_2021_shared_task
        and extracts the kpm_data directory.
        
        Args:
            splits: Dataset splits to load (default: ("dev", "train"))
            
        Returns:
            query_to_label_series: Dictionary mapping key points to pandas Series of labels
        """        
        data_dir = os.path.join(DATA_OUT_DIR, "kpm")
        kpm_data_dir = os.path.join(data_dir, "kpm_data")
        os.makedirs(data_dir, exist_ok=True)
        
        if not os.path.exists(kpm_data_dir):
            # Download the repository as a zip file
            zip_path = os.path.join(data_dir, "KPA_2021_shared_task.zip")
            url = "https://github.com/IBM/KPA_2021_shared_task/archive/refs/heads/main.zip"
            
            print(f"Downloading ArgKP21 dataset from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract the zip file
            print("Extracting ArgKP21 dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Move kpm_data to the expected location
            extracted_dir = os.path.join(data_dir, "KPA_2021_shared_task-main")
            extracted_kpm_data = os.path.join(extracted_dir, "kpm_data")
            
            if os.path.exists(extracted_kpm_data):
                shutil.move(extracted_kpm_data, kpm_data_dir)
            
            # Clean up
            os.remove(zip_path)
            if os.path.exists(extracted_dir):
                shutil.rmtree(extracted_dir)
            
            print("ArgKP21 dataset downloaded and extracted successfully.")
        
        query_to_rows = defaultdict(list)
        for split in splits:
            arguments = pd.read_csv(os.path.join(kpm_data_dir, f"arguments_{split}.csv"))
            key_points = pd.read_csv(os.path.join(kpm_data_dir, f"key_points_{split}.csv"))
            labels = pd.read_csv(os.path.join(kpm_data_dir, f"labels_{split}.csv"))

            for arg_id, key_point_id, label in zip(labels["arg_id"], labels["key_point_id"], labels["label"]):
                arg = arguments.query("arg_id == @arg_id")['argument'].values[0]
                key_point = key_points.query("key_point_id == @key_point_id")['key_point'].values[0]
                query_to_rows[key_point].append({"argument": arg, "label": label})

        query_to_label_series = {query: pd.DataFrame(rows).set_index("argument")["label"]
                                 for query, rows in query_to_rows.items()}
        return query_to_label_series


class RealScholarQuery(Dataset):
    dataset_name = "RealScholarQuery"
    corpus_size_limit = 25000

    def load_gold(self):
        data_dir = os.path.join(DATA_OUT_DIR, "pasa_dataset")
        os.makedirs(data_dir, exist_ok=True)
        snapshot_download(
            local_dir=data_dir,
            repo_id="CarlanLark/pasa-dataset",
            etag_timeout=30,
            repo_type="dataset",
        )

        paper_zip = zipfile.ZipFile(
            f"{data_dir}/paper_database/cs_paper_2nd.zip", "r")
        arxiv_id_to_title = json.load(
            open(f"{data_dir}/paper_database/id2paper.json"))
        title_to_arxiv_id = {v: k for k, v in arxiv_id_to_title.items()}
        all_data = {}
        for filename in tqdm(paper_zip.filelist, desc="Reading paper jsons"):
            paper_data = json.loads(paper_zip.open(filename).read().decode("utf-8"))
            paper_data['arxiv_id'] = title_to_arxiv_id[paper_data['title']]
            all_data[paper_data['arxiv_id']] = paper_data

        with open(f"{data_dir}/RealScholarQuery/test.jsonl") as f:
            lines = f.readlines()
        test_data = [json.loads(l) for l in lines]
        benchmark_df = pd.DataFrame(test_data)

        if self.corpus_size_limit and self.corpus_size_limit < len(all_data.keys()):
            # dataset is huge so we can choose to downsample, but always keep the positive docs
            doc_ids_in_gold = {doc_id for answer_doc_ids in benchmark_df['answer_arxiv_id'].values
                               for doc_id in answer_doc_ids if doc_id in all_data}
            other_doc_ids = set(all_data.keys()).difference(doc_ids_in_gold)
            sampled_doc_ids = random.Random(42).sample(sorted(other_doc_ids), self.corpus_size_limit - len(doc_ids_in_gold))
            all_doc_ids = list(doc_ids_in_gold) + sampled_doc_ids
        else:
            all_doc_ids = list(all_data.keys())

        query_to_label_series = {}
        for question, gold_ids in tqdm(zip(benchmark_df['question'], benchmark_df['answer_arxiv_id']),
                                       desc="Building query label series", total=len(benchmark_df)):
            question_rows = [
                {"text": all_data[doc_id]["abstract"],
                 "label": 1 if doc_id in gold_ids else 0}
                for doc_id in all_doc_ids if doc_id in all_data
            ]
            query_to_label_series[question] = pd.DataFrame(question_rows).set_index(["text"])["label"]

        return query_to_label_series


class FollowIR(Dataset):
    dataset_name = "FollowIR"

    def load_gold(self):
        from mteb.abstasks.retrieval_dataset_loaders import RetrievalDatasetLoader
        from mteb.tasks.instruction_reranking.eng import Core17InstructionRetrieval, Robust04InstructionRetrieval, \
            News21InstructionRetrieval

        query_to_label_series = {}
        for task in [Core17InstructionRetrieval, Robust04InstructionRetrieval, News21InstructionRetrieval]:
            ds = RetrievalDatasetLoader(
                hf_repo=task.metadata.dataset["path"],
                revision=task.metadata.dataset["revision"],
            ).load()

            queries_dict = dict(zip(ds["queries"]["id"], zip(ds["queries"]["text"], ds["queries"]["instruction"])))
            corpus_dict = dict(zip(ds["corpus"]["id"], zip(ds["corpus"]["title"], ds["corpus"]["text"])))

            long_df = load_dataset(task.metadata.dataset["path"])["test"].to_pandas()
            long_df["query"] = long_df["query-id"].apply(lambda qid: "\n".join(queries_dict[qid]))
            long_df["text"] = long_df["corpus-id"].apply(lambda cid: "\n".join(corpus_dict[cid]))
            subset_dict = {query: (query_df["score"] > 0).astype(int)  # TODO use 1 vs. 2 labels?
                           for query, query_df in long_df.set_index(["text"]).groupby("query")}
            query_to_label_series.update(subset_dict)
        return query_to_label_series


class BEIRDataset(Dataset):
    """Base class for BEIR datasets with common loading logic."""
    dataset_name: str = None  # Must be overridden
    beir_dataset_name: str = None  # Must be overridden (e.g., "scifact", "trec-covid")
    corpus_size_limit: int = 5000
    
    def load_gold(self, split="test"):
        """
        Load BEIR dataset with common logic.
        
        Args:
            split: Dataset split to load (default: "test")
            
        Returns:
            query_to_label_series: Dictionary mapping queries to pandas Series of labels
        """
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        # Download and load dataset
        data_path = util.download_and_unzip(
            url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.beir_dataset_name}.zip",
            out_dir=DATA_OUT_DIR
        )
        
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        
        # Subsample corpus if it's too large, but always keep positive documents
        if self.corpus_size_limit and self.corpus_size_limit < len(corpus):
            # Collect all document IDs that have positive relevance judgments
            doc_ids_in_gold = {doc_id for query_rels in qrels.values()
                               for doc_id, rel in query_rels.items() if rel > 0 and doc_id in corpus}

            other_doc_ids = set(corpus.keys()).difference(doc_ids_in_gold)
            # Sample from other docs to reach corpus_size_limit
            sampled_doc_ids = random.Random(42).sample(
                sorted(other_doc_ids),
                self.corpus_size_limit - len(doc_ids_in_gold)
            )
            
            selected_doc_ids = list(doc_ids_in_gold) + sampled_doc_ids
        else:
            selected_doc_ids = list(corpus.keys())
        
        query_to_label_series = {}
        for query_id, query_text in tqdm(queries.items(), desc=f"Processing {self.dataset_name} queries"):
            if query_id not in qrels:
                continue
                
            # Get relevance judgments for this query
            doc_relevances = qrels[query_id]
            
            # Create rows only for selected documents
            query_rows = []
            for doc_id in selected_doc_ids:
                doc_content = corpus[doc_id]
                doc_text = doc_content.get("title", "") + " " + doc_content.get("text", "")
                
                # Get relevance label (default to 0 if not in qrels)
                label = doc_relevances.get(doc_id, 0)
                
                query_rows.append({
                    "text": doc_text.strip(),
                    "label": 1 if label > 0 else 0  # Binary relevance
                })
            
            # Create pandas Series indexed by document text
            query_df = pd.DataFrame(query_rows).set_index("text")
            query_to_label_series[query_text] = query_df["label"]
        
        return query_to_label_series


class NFCorpus(BEIRDataset):
    dataset_name = "NFCorpus"
    beir_dataset_name = "nfcorpus"
    corpus_size_limit = None


class Banking77(Dataset):
    dataset_name = "Banking77"

    def load_gold(self, splits=("train", "test")):
        """
        Load Banking77 dataset from HuggingFace.

        This is a banking intent classification dataset with 77 fine-grained intents.
        Each intent (class) is treated as a query, and the texts are documents.

        Args:
            splits: Dataset splits to load (default: ("train", "test"))

        Returns:
            query_to_label_series: Dictionary mapping intents to pandas Series of labels
        """
        dataset = load_dataset("mteb/banking77")

        # Combine specified splits
        all_data = []
        for split in splits:
            if split in dataset:
                split_df = dataset[split].to_pandas()
                all_data.append(split_df)

        df = pd.concat(all_data, ignore_index=True)

        # Get all unique label texts (intents)
        all_labels = df['label_text'].unique()

        # Create full cartesian product (text × labels)
        texts_df = df[['text']].drop_duplicates()
        labels_df = pd.DataFrame({'label_text': all_labels})

        full_df = texts_df.merge(labels_df, how='cross')

        # Merge with actual labels
        result_df = full_df.merge(
            df[['text', 'label_text']].assign(actual_label=1),
            on=['text', 'label_text'],
            how='left'
        ).fillna({'actual_label': 0}).astype({'actual_label': int})

        # Set text as index
        result_df = result_df.set_index('text')

        # Group by label-text (intent) and create series indexed by text
        query_to_label_series = {
            label.replace("_", " "): label_df['actual_label']
            for label, label_df in result_df.groupby('label_text')
        }

        return query_to_label_series


class Clinc150(Dataset):
    dataset_name = "Clinc150"

    def load_gold(self):
        """
        Load CLINC150 dataset from Github.

        This is an intent classification dataset with 150 intent classes
        plus an out-of-scope class. Each intent has multiple utterances.

        Intent names can be customized by filling out the 'new_intent_name' column
        in the intent_renaming_new.csv file in the data/clinc150/ directory.

        Returns:
            query_to_label_series: Dictionary mapping intents to pandas Series of labels
        """
        # Setup data directory
        data_dir = os.path.join(DATA_OUT_DIR, "clinc150")
        os.makedirs(data_dir, exist_ok=True)

        # Check if JSON file exists, download if not
        json_path = os.path.join(data_dir, "clinc150_full.json")
        
        if not os.path.exists(json_path):
            # Download the data_full.json file from GitHub
            url = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"
            print(f"Downloading CLINC150 dataset from {url}...")
            urllib.request.urlretrieve(url, json_path)
            print("CLINC150 dataset downloaded successfully.")

        # Load intent renaming mapping
        intent_mapping = {}
        renaming_csv_path = os.path.join(data_dir, "clinc150_intent_mapping.csv")
        renaming_df = pd.read_csv(renaming_csv_path)
        # Only use mappings where new_intent_name is not empty
        for _, row in renaming_df.iterrows():
            intent_mapping[row['intent_name']] = row['intent_description'].strip()

        # Load data from JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Extract only the 'train' section
        train_data = data['train']

        # Convert to list of dicts, applying intent renaming if available
        all_data = []
        for text, intent in train_data:
            # Apply intent mapping if available, otherwise use original with underscores replaced
            mapped_intent = intent_mapping[intent]
            all_data.append({"text": text, "intent": mapped_intent})

        df = pd.DataFrame(all_data)

        # Filter out "oos" (out-of-scope) intent as it's not a real intent
        df = df[df['intent'] != 'oos']

        # Get all unique intents and texts
        all_intents = df['intent'].unique()
        all_texts = df['text'].unique()

        # Create full cartesian product (text × intent)
        texts_df = pd.DataFrame({'text': all_texts})
        intents_df = pd.DataFrame({'intent': all_intents})
        
        full = texts_df.merge(intents_df, how='cross')

        # Create positive labels from original data
        positive_labels = df[['text', 'intent']].copy()
        positive_labels['label'] = 1

        # Merge and fill missing as 0
        result = full.merge(
            positive_labels,
            on=['text', 'intent'],
            how='left'
        ).fillna({'label': 0}).astype({'label': int})

        # Clean up intent names (replace underscores with spaces for better readability)
        result['intent'] = result['intent'].str.replace('_', ' ')

        # Set index and group by intent
        result = result.set_index(['text'])
        query_to_label_series = {
            intent: intent_df['label']
            for intent, intent_df in result.groupby('intent')
        }

        return query_to_label_series


@dataclass
class Datasets:
    ArgKP21 = ArgKP21
    RealScholarQuery = RealScholarQuery
    FollowIR = FollowIR
    NFCorpus = NFCorpus
    Banking77 = Banking77
    Clinc150 = Clinc150

    @staticmethod
    def all_datasets():
        return [var for var in vars(Datasets)
                if isinstance(getattr(Datasets, var), type)]
