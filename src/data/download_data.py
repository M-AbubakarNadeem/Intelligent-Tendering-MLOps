"""
Download PubMed RCT dataset for sequential sentence classification.

The PubMed RCT dataset contains medical abstracts where each sentence
is labeled with its role (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS).
This mirrors the RFQ document classification task from the paper where
each position in a document is classified with a product set.

Source: https://github.com/Franck-Dernoncourt/pubmed-rct
"""

import os
import requests
import argparse
from pathlib import Path


# PubMed 20k RCT dataset URLs (smaller version for faster experiments)
BASE_URL = (
    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/"
    "PubMed_20k_RCT_numbers_replaced_with_at_sign"
)

FILES = {
    "train.txt": f"{BASE_URL}/train.txt",
    "dev.txt": f"{BASE_URL}/dev.txt",
    "test.txt": f"{BASE_URL}/test.txt",
}


def download_file(url: str, save_path: str) -> None:
    """Download a file from URL to local path."""
    print(f"Downloading {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    print(f"  Saved to {save_path} ({len(response.text):,} chars)")


def download_dataset(data_dir: str = "data/raw") -> None:
    """Download all PubMed RCT dataset files."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading PubMed 20k RCT Dataset")
    print("=" * 60)

    for filename, url in FILES.items():
        save_path = data_path / filename
        if save_path.exists():
            print(f"  {filename} already exists, skipping.")
            continue
        download_file(url, str(save_path))

    print("\nDataset download complete!")
    print(f"Files saved to: {data_path.resolve()}")


def verify_dataset(data_dir: str = "data/raw") -> dict:
    """Verify the downloaded dataset and print statistics."""
    data_path = Path(data_dir)
    stats = {}

    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    for filename in FILES.keys():
        filepath = data_path / filename
        if not filepath.exists():
            print(f"  WARNING: {filename} not found!")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        n_abstracts = sum(1 for line in lines if line.startswith("###"))
        n_sentences = sum(
            1
            for line in lines
            if line.strip()
            and not line.startswith("###")
            and "\t" in line
        )

        # Count labels
        label_counts = {}
        for line in lines:
            line = line.strip()
            if line and not line.startswith("###") and "\t" in line:
                label = line.split("\t")[0]
                label_counts[label] = label_counts.get(label, 0) + 1

        stats[filename] = {
            "abstracts": n_abstracts,
            "sentences": n_sentences,
            "labels": label_counts,
        }

        print(f"\n  {filename}:")
        print(f"    Abstracts: {n_abstracts:,}")
        print(f"    Sentences: {n_sentences:,}")
        print(f"    Labels: {label_counts}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PubMed RCT dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Directory to save dataset files",
    )
    args = parser.parse_args()

    download_dataset(args.data_dir)
    verify_dataset(args.data_dir)
