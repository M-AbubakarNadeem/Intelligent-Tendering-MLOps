"""
Preprocess PubMed RCT dataset for both SinSent and SeqSent models.

Creates two types of DataLoaders:
1. SinSent: Each sentence is an independent sample (standard classification)
2. SeqSent: Each abstract is a sample with sequence of sentences (sequential classification)

This mirrors the paper's two approaches:
- SinSent classifies each RFQ position independently
- SeqSent classifies all positions jointly using surrounding context
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Tuple


# Label mapping for PubMed RCT
LABEL_MAP = {
    "BACKGROUND": 0,
    "OBJECTIVE": 1,
    "METHODS": 2,
    "RESULTS": 3,
    "CONCLUSIONS": 4,
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)


def parse_pubmed_rct(filepath: str) -> List[Dict]:
    """
    Parse PubMed RCT format into list of abstracts.

    Each abstract is a dict with:
        - 'id': abstract ID
        - 'sentences': list of sentence texts
        - 'labels': list of label strings
    """
    abstracts = []
    current_abstract = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("###"):
                # New abstract
                if current_abstract and current_abstract["sentences"]:
                    abstracts.append(current_abstract)
                current_abstract = {
                    "id": line.replace("###", "").strip(),
                    "sentences": [],
                    "labels": [],
                }
            elif "\t" in line and current_abstract is not None:
                parts = line.split("\t", 1)
                if len(parts) == 2 and parts[0] in LABEL_MAP:
                    current_abstract["labels"].append(parts[0])
                    current_abstract["sentences"].append(parts[1])

    # Don't forget last abstract
    if current_abstract and current_abstract["sentences"]:
        abstracts.append(current_abstract)

    return abstracts


class SinSentDataset(Dataset):
    """
    Single Sentence Dataset — each sentence is classified independently.
    Corresponds to the SinSent model in the paper.
    """

    def __init__(
        self,
        abstracts: List[Dict],
        tokenizer: BertTokenizer,
        max_len: int = 128,
    ):
        self.sentences = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Flatten abstracts into individual sentences
        for abstract in abstracts:
            for sent, label in zip(abstract["sentences"], abstract["labels"]):
                self.sentences.append(sent)
                self.labels.append(LABEL_MAP[label])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class SeqSentDataset(Dataset):
    """
    Sequential Sentence Dataset — each abstract is a sequence of sentences.
    Corresponds to the SeqSent model in the paper.

    Each sample contains all sentences from one abstract, enabling
    the model to leverage surrounding context for classification.
    """

    def __init__(
        self,
        abstracts: List[Dict],
        tokenizer: BertTokenizer,
        max_len: int = 128,
        max_sentences: int = 30,
    ):
        self.abstracts = abstracts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract = self.abstracts[idx]
        sentences = abstract["sentences"][: self.max_sentences]
        labels = abstract["labels"][: self.max_sentences]

        # Tokenize each sentence
        input_ids_list = []
        attention_mask_list = []

        for sent in sentences:
            encoding = self.tokenizer(
                sent,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(encoding["input_ids"].squeeze(0))
            attention_mask_list.append(encoding["attention_mask"].squeeze(0))

        # Convert labels
        label_ids = [LABEL_MAP[label] for label in labels]
        num_sentences = len(sentences)

        return {
            "input_ids": torch.stack(input_ids_list),  # (num_sent, max_len)
            "attention_mask": torch.stack(attention_mask_list),  # (num_sent, max_len)
            "labels": torch.tensor(label_ids, dtype=torch.long),  # (num_sent,)
            "num_sentences": num_sentences,
        }


def seqsent_collate_fn(batch):
    """
    Custom collate function for SeqSent that handles variable-length
    sequences of sentences (different abstracts have different lengths).
    """
    max_num_sent = max(item["num_sentences"] for item in batch)
    max_len = batch[0]["input_ids"].shape[1]

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_num_sentences = []

    for item in batch:
        num_sent = item["num_sentences"]
        pad_size = max_num_sent - num_sent

        # Pad input_ids and attention_mask
        if pad_size > 0:
            pad_input = torch.zeros(pad_size, max_len, dtype=torch.long)
            pad_mask = torch.zeros(pad_size, max_len, dtype=torch.long)
            pad_labels = torch.full((pad_size,), -100, dtype=torch.long)  # ignore index

            input_ids = torch.cat([item["input_ids"], pad_input], dim=0)
            attention_mask = torch.cat([item["attention_mask"], pad_mask], dim=0)
            labels = torch.cat([item["labels"], pad_labels], dim=0)
        else:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)
        batch_num_sentences.append(item["num_sentences"])

    return {
        "input_ids": torch.stack(batch_input_ids),  # (batch, max_sent, max_len)
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),  # (batch, max_sent)
        "num_sentences": torch.tensor(batch_num_sentences),
    }


def create_dataloaders(
    data_dir: str = "data/raw",
    model_type: str = "sinsent",
    tokenizer_name: str = "bert-base-uncased",
    max_len: int = 128,
    batch_size: int = 32,
    max_samples: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for the specified model type.

    Args:
        data_dir: Path to raw data directory
        model_type: 'sinsent' or 'seqsent'
        tokenizer_name: HuggingFace tokenizer name
        max_len: Maximum token length per sentence
        batch_size: Batch size
        max_samples: Limit number of abstracts (for development)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

    # Parse data files
    train_abstracts = parse_pubmed_rct(os.path.join(data_dir, "train.txt"))
    val_abstracts = parse_pubmed_rct(os.path.join(data_dir, "dev.txt"))
    test_abstracts = parse_pubmed_rct(os.path.join(data_dir, "test.txt"))

    # Limit samples for development
    if max_samples:
        train_abstracts = train_abstracts[:max_samples]
        val_abstracts = val_abstracts[: max(max_samples // 5, 50)]
        test_abstracts = test_abstracts[: max(max_samples // 5, 50)]

    print("\nDataset sizes:")
    print(f"  Train: {len(train_abstracts)} abstracts")
    print(f"  Val:   {len(val_abstracts)} abstracts")
    print(f"  Test:  {len(test_abstracts)} abstracts")

    if model_type == "sinsent":
        train_dataset = SinSentDataset(train_abstracts, tokenizer, max_len)
        val_dataset = SinSentDataset(val_abstracts, tokenizer, max_len)
        test_dataset = SinSentDataset(test_abstracts, tokenizer, max_len)

        print("\nSinSent samples:")
        print(f"  Train: {len(train_dataset)} sentences")
        print(f"  Val:   {len(val_dataset)} sentences")
        print(f"  Test:  {len(test_dataset)} sentences")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    elif model_type == "seqsent":
        train_dataset = SeqSentDataset(train_abstracts, tokenizer, max_len)
        val_dataset = SeqSentDataset(val_abstracts, tokenizer, max_len)
        test_dataset = SeqSentDataset(test_abstracts, tokenizer, max_len)

        print("\nSeqSent samples:")
        print(f"  Train: {len(train_dataset)} abstracts")
        print(f"  Val:   {len(val_dataset)} abstracts")
        print(f"  Test:  {len(test_dataset)} abstracts")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=seqsent_collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=seqsent_collate_fn,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=seqsent_collate_fn,
            num_workers=0,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return train_loader, val_loader, test_loader


def save_label_map(output_dir: str = "data/processed"):
    """Save label mapping for later use."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump({"label_to_id": LABEL_MAP, "id_to_label": ID_TO_LABEL}, f, indent=2)
    print(f"Label map saved to {output_dir}/label_map.json")


if __name__ == "__main__":
    # Test data loading
    save_label_map()

    print("\n--- Testing SinSent DataLoader ---")
    train_loader, val_loader, test_loader = create_dataloaders(
        model_type="sinsent", batch_size=4, max_samples=10
    )
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")

    print("\n--- Testing SeqSent DataLoader ---")
    train_loader, val_loader, test_loader = create_dataloaders(
        model_type="seqsent", batch_size=2, max_samples=10
    )
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
    print(f"Batch num_sentences: {batch['num_sentences']}")
