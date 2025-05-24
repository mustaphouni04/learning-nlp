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

def process_serialized(data: List[Dict[str, str]], train: bool, test_size: float):
    texts    = [str(rec.get("Text", ""))    for rec in data]
    summaries= [str(rec.get("Summary", "")) for rec in data]
    if train:
        train_texts, _, train_summaries, _ = train_test_split(texts, summaries, test_size=test_size, random_state=42)
        pairs = (train_texts, train_summaries) 
        return pairs
    else:
        _, test_texts, _, test_summaries = train_test_split(texts, summaries, test_size=test_size, random_state=42)
        pairs = (test_texts, test_summaries) 
        return pairs

def split_dataset(dataset: Union[Path,str] | List[Dict[str, str]], train: bool, test_size: float):
    if isinstance(dataset, str) or isinstance(dataset, Path):
        dataset = str(dataset)
        with open(dataset, "r") as f:
            file = json.load(f)
            return process_serialized(file, train, test_size)
    elif isinstance(dataset, list):
        return process_serialized(dataset, train, test_size)
    else:
        raise TypeError("Invalid dataset type")

@dataclass
class Pipeline(Dataset):
    dataset: Union[Path,str] | List[Dict[str, str]]
    train: bool = True
    test_size: float = 0.1
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    max_length: int = 1024
    
    def __post_init__(self):
        pairs = split_dataset(self.dataset, self.train, self.test_size) 
        self.texts, self.summaries = pairs

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        t = "Generate summary: \n" + self.texts[idx]
        s = self.summaries[idx]

        enc = self.tokenizer(
                t, text_pair=None,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                )

        label_enc = self.tokenizer(
                s, padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                )

        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": label_enc.input_ids.squeeze(0)
        }

 
