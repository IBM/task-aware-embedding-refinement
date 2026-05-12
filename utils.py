import hashlib

import torch
import json
import os
import yaml
from datetime import datetime


def get_device():
    """Detect and return the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return "mps"  # mac GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
    

def hash_text(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_with_imap(func):
    def inner(self, args):
        return func(self, *args)
    return inner


def save_experiment_config(config_dict, output_dir):
    """Save experiment configuration to JSON file."""
    config = config_dict.copy()
    config['timestamp'] = datetime.now().isoformat()
    
    config_path = os.path.join(output_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved experiment configuration to {config_path}")
    return config_path


def load_instruction_templates(yaml_path):
    """Load instruction templates from YAML file."""
    with open(yaml_path, 'r') as f:
        templates = yaml.safe_load(f)
    return templates


def get_templates_for_model(model_name, templates_dict, dataset_name=None):
    """Get query and document templates for a given model and dataset."""

    templates = templates_dict['model_templates'].get(
        model_name, templates_dict['model_templates']['Default'])

    # add dataset instructions
    dataset_config = templates_dict['dataset_instructions'][dataset_name]
    if 'query_template' in dataset_config:  # override instruction template
        templates['query_template'] = dataset_config['query_template']
    else:
        templates['query_template'] = templates['query_template'].replace("{instruction}",
                                                                          dataset_config['instruction'])

    return templates['query_template'], templates['document_template']


def get_llm_task_instruction(templates_dict, dataset_name):
    """Get task instruction for a given dataset."""
    dataset_config = templates_dict['dataset_instructions'][dataset_name]
    if dataset_config['task_type'] == 'classification':
        return dataset_config['instruction']
    else:
        return ""


def apply_template(texts, template):
    """Apply instruction template to texts."""
    if isinstance(texts, str):
        return template.format(text=texts)
    return [template.format(text=text) for text in texts]


def slugify(text: str, max_len: int = 128) -> str:
    slug = "".join([c if (c.isalnum() or c in "-_.") else "_" for c in text]).strip("_")
    return slug[:max_len]
