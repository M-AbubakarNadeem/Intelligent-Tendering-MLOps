"""
Single Sentence Classifier (SinSent)

Classifies each sentence/position independently using BERT + classification head.
This corresponds to the SinSent model from the paper:
"Fine-tuned the GBERT model for sentence classification by adding a
classification head and use the [CLS] token to predict the class labels
for each position text independently."

Architecture:
    Input sentence → BERT → [CLS] token → Dropout → Linear → Class logits
"""

import torch.nn as nn
from transformers import BertModel


class SingleSentenceClassifier(nn.Module):
    """
    BERT-based single sentence classifier.

    Each sentence is classified independently without any context
    from surrounding sentences in the document.

    Paper config: 50 epochs, batch_size=32, lr=1e-05, dropout=0.3
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)  # nosec B615
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size,) optional

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def get_num_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
