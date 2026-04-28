"""
Unit tests for model architectures.
"""

import sys
import pytest
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.single_sentence import SingleSentenceClassifier
from src.models.sequential_sentence import SequentialSentenceClassifier
from src.data.preprocess import NUM_LABELS


class TestSingleSentenceClassifier:
    """Test SinSent model architecture."""

    @pytest.fixture
    def model(self):
        return SingleSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name="bert-base-uncased",
            dropout=0.3,
        )

    def test_forward_shape(self, model):
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        outputs = model(input_ids, attention_mask)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, NUM_LABELS)

    def test_forward_with_labels(self, model):
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, NUM_LABELS, (batch_size,))

        outputs = model(input_ids, attention_mask, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].dim() == 0  # scalar

    def test_parameter_count(self, model):
        params = model.get_num_parameters()
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["trainable"] <= params["total"]


class TestSequentialSentenceClassifier:
    """Test SeqSent model architecture."""

    @pytest.fixture
    def model(self):
        return SequentialSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name="bert-base-uncased",
            dropout=0.1,
        )

    def test_forward_shape(self, model):
        batch_size = 2
        num_sent = 5
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, num_sent, seq_len))
        attention_mask = torch.ones(batch_size, num_sent, seq_len, dtype=torch.long)
        num_sentences = torch.tensor([5, 3])

        outputs = model(input_ids, attention_mask, num_sentences=num_sentences)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, num_sent, NUM_LABELS)

    def test_forward_with_labels(self, model):
        batch_size = 2
        num_sent = 5
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, num_sent, seq_len))
        attention_mask = torch.ones(batch_size, num_sent, seq_len, dtype=torch.long)
        labels = torch.randint(0, NUM_LABELS, (batch_size, num_sent))
        # Set padding to -100
        labels[1, 3:] = -100
        num_sentences = torch.tensor([5, 3])

        outputs = model(
            input_ids, attention_mask, labels=labels, num_sentences=num_sentences
        )
        assert "loss" in outputs
        assert outputs["loss"].dim() == 0

    def test_seqsent_has_more_params(self, model):
        sinsent = SingleSentenceClassifier(NUM_LABELS)
        sin_params = sinsent.get_num_parameters()["total"]
        seq_params = model.get_num_parameters()["total"]
        # SeqSent should have more params due to BiLSTM
        assert seq_params > sin_params


class TestModelComparison:
    """Test that both models can process the same data differently."""

    def test_different_outputs_for_same_input(self):
        """SinSent and SeqSent should give different shaped outputs."""
        sinsent = SingleSentenceClassifier(NUM_LABELS)
        seqsent = SequentialSentenceClassifier(NUM_LABELS)

        # Single sentence input for SinSent
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        sin_out = sinsent(input_ids, attention_mask)

        # Sequence input for SeqSent
        seq_input_ids = torch.randint(0, 1000, (1, 3, 32))
        seq_attention_mask = torch.ones(1, 3, 32, dtype=torch.long)
        seq_out = seqsent(seq_input_ids, seq_attention_mask)

        assert sin_out["logits"].shape == (1, NUM_LABELS)  # per sentence
        assert seq_out["logits"].shape == (1, 3, NUM_LABELS)  # per sentence in sequence
