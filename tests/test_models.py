import sys
from pathlib import Path
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.single_sentence import SingleSentenceClassifier  # noqa: E402
from src.models.sequential_sentence import SequentialSentenceClassifier  # noqa: E402
from src.data.preprocess import NUM_LABELS  # noqa: E402


def test_sinsent_model():
    model = SingleSentenceClassifier(
        num_labels=NUM_LABELS, model_name="prajjwal1/bert-tiny"
    )
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones(2, 16)
    outputs = model(input_ids, attention_mask)
    assert outputs.shape == (2, NUM_LABELS)


def test_seqsent_model():
    model = SequentialSentenceClassifier(
        num_labels=NUM_LABELS, model_name="prajjwal1/bert-tiny"
    )
    # (batch, max_sent, seq_len)
    input_ids = torch.randint(0, 100, (2, 3, 16))
    attention_mask = torch.ones(2, 3, 16)
    num_sentences = torch.tensor([3, 2])
    outputs = model(input_ids, attention_mask, num_sentences=num_sentences)
    assert outputs.shape == (2, 3, NUM_LABELS)
