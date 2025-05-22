from torch import nn
from transformers import AutoTokenizer
from pathlib import Path
from typing import Dict, Union, List
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import json

class ByT5Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, 
                input_ids,
                attention_mask,
                labels):
        outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels)
        return outputs.loss

class ByT5ModelOutput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, 
                input_ids,
                attention_mask):
        outputs = self.model(
                input_ids = input_ids,
                attention_mask = attention_mask)
        return outputs

def process_serialized(data: List[Dict[str, str]], train: bool):
    texts = [rec["Text"] for rec in data]
    summaries = [rec["Summary"] for rec in data]
    if train:
        train_texts, _, train_summaries, _ = train_test_split(texts, summaries, test_size=0.2, random_state=42)
        pairs = (train_texts, train_summaries) 
        return pairs
    else:
        _, test_texts, _, test_summaries = train_test_split(texts, summaries, test_size=0.2, random_state=42)
        pairs = (test_texts, test_summaries) 
        return pairs

def split_dataset(dataset: Union[Path,str] | List[Dict[str, str]], train: bool):
    if isinstance(dataset, str) or isinstance(dataset, Path):
        dataset = str(dataset)
        with open(dataset, "r") as f:
            file = json.load(f)
            return process_serialized(file, train)
    elif isinstance(dataset, dict):
        return process_serialized(dataset, train)
    else:
        raise TypeError("Invalid dataset type")

@dataclass
class Pipeline(Dataset):
    dataset: Union[Path,str] | List[Dict[str, str]]
    train: bool = True
    
    def __post_init__(self):
        pairs = split_dataset(self.dataset, self.train) 
        tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

        self.inputs = tokenizer(
            pairs[0], padding=True, truncation=True, return_tensors="pt"
        )
        self.labels = tokenizer(
            pairs[1], padding=True, truncation=True, return_tensors="pt"
        ).input_ids

    def __len__(self):
        return self.inputs["input_ids"].size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.labels[idx]
        }

 
