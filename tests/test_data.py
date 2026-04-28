"""
Unit tests for data pipeline.
"""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import (
    parse_pubmed_rct,
    LABEL_MAP,
    ID_TO_LABEL,
    NUM_LABELS,
    SinSentDataset,
    SeqSentDataset,
)


class TestLabelMapping:
    """Test label constants."""

    def test_num_labels(self):
        assert NUM_LABELS == 5

    def test_label_map_keys(self):
        expected = {"BACKGROUND", "OBJECTIVE", "METHODS", "RESULTS", "CONCLUSIONS"}
        assert set(LABEL_MAP.keys()) == expected

    def test_id_to_label_inverse(self):
        for label, idx in LABEL_MAP.items():
            assert ID_TO_LABEL[idx] == label

    def test_label_ids_are_sequential(self):
        ids = sorted(LABEL_MAP.values())
        assert ids == list(range(NUM_LABELS))


class TestParsePubmedRCT:
    """Test data parsing (requires downloaded data)."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a sample data file for testing."""
        content = """###12345678
BACKGROUND\tThis is a background sentence.
BACKGROUND\tAnother background sentence.
OBJECTIVE\tThe objective of this study.
METHODS\tWe used these methods.
RESULTS\tThe results showed improvement.
CONCLUSIONS\tIn conclusion, we found that.

###87654321
BACKGROUND\tDifferent study background.
METHODS\tDifferent methods were used.
RESULTS\tResults were significant.
CONCLUSIONS\tFinal conclusions here.
"""
        filepath = tmp_path / "test_data.txt"
        filepath.write_text(content, encoding="utf-8")
        return str(filepath)

    def test_parse_returns_list(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        assert isinstance(abstracts, list)

    def test_parse_correct_count(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        assert len(abstracts) == 2

    def test_parse_abstract_structure(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        abstract = abstracts[0]
        assert "id" in abstract
        assert "sentences" in abstract
        assert "labels" in abstract

    def test_parse_first_abstract(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        first = abstracts[0]
        assert first["id"] == "12345678"
        assert len(first["sentences"]) == 6
        assert len(first["labels"]) == 6
        assert first["labels"][0] == "BACKGROUND"
        assert first["labels"][-1] == "CONCLUSIONS"

    def test_parse_second_abstract(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        second = abstracts[1]
        assert second["id"] == "87654321"
        assert len(second["sentences"]) == 4

    def test_all_labels_valid(self, sample_data_file):
        abstracts = parse_pubmed_rct(sample_data_file)
        for abstract in abstracts:
            for label in abstract["labels"]:
                assert label in LABEL_MAP, f"Invalid label: {label}"


class TestSinSentDataset:
    """Test SinSent dataset."""

    @pytest.fixture
    def sample_abstracts(self):
        return [
            {
                "id": "1",
                "sentences": ["Sentence one.", "Sentence two."],
                "labels": ["BACKGROUND", "METHODS"],
            },
            {
                "id": "2",
                "sentences": ["Sentence three."],
                "labels": ["RESULTS"],
            },
        ]

    def test_sinsent_flattens_sentences(self, sample_abstracts):
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = SinSentDataset(sample_abstracts, tokenizer, max_len=32)
        assert len(dataset) == 3  # 2 + 1 sentences total

    def test_sinsent_item_structure(self, sample_abstracts):
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = SinSentDataset(sample_abstracts, tokenizer, max_len=32)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "label" in item
        assert item["input_ids"].shape[0] == 32  # max_len


class TestSeqSentDataset:
    """Test SeqSent dataset."""

    @pytest.fixture
    def sample_abstracts(self):
        return [
            {
                "id": "1",
                "sentences": ["Sentence one.", "Sentence two.", "Sentence three."],
                "labels": ["BACKGROUND", "METHODS", "RESULTS"],
            },
        ]

    def test_seqsent_per_abstract(self, sample_abstracts):
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = SeqSentDataset(sample_abstracts, tokenizer, max_len=32)
        assert len(dataset) == 1  # 1 abstract

    def test_seqsent_item_structure(self, sample_abstracts):
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        dataset = SeqSentDataset(sample_abstracts, tokenizer, max_len=32)
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert "num_sentences" in item
        assert item["input_ids"].shape == (3, 32)  # (num_sent, max_len)
        assert item["labels"].shape == (3,)
        assert item["num_sentences"] == 3
