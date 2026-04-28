import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import parse_pubmed_rct, SinSentDataset  # noqa: E402


def test_parse_pubmed_rct(tmp_path):
    # Create a small dummy file
    content = "### 123\nMETHODS\tSentence 1.\nRESULTS\tSentence 2.\n"
    d = tmp_path / "sub"
    d.mkdir()
    f = d / "test.txt"
    f.write_text(content)

    abstracts = parse_pubmed_rct(str(f))
    assert len(abstracts) == 1
    assert abstracts[0]["id"] == "123"
    assert len(abstracts[0]["sentences"]) == 2


def test_sinsent_dataset():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    abstracts = [
        {"id": "1", "sentences": ["Sent 1", "Sent 2"], "labels": ["METHODS", "RESULTS"]}
    ]
    dataset = SinSentDataset(abstracts, tokenizer, max_len=16)

    assert len(dataset) == 2
    item = dataset[0]
    assert "input_ids" in item
    assert "label" in item
