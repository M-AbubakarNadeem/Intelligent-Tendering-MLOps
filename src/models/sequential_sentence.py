"""
Sequential Sentence Classifier (SeqSent)

Classifies all sentences in a document jointly, leveraging surrounding context.
This corresponds to the SeqSent model from the paper:
"All positions are classified jointly and in context of their surrounding
request using a sequential sentence classification model."

Architecture:
    For each sentence: BERT → [CLS] embedding
    Sequence of embeddings → BiLSTM → Dropout → Linear → Class logits per sentence

The BiLSTM captures sequential dependencies between sentences,
enabling the model to recognize document-level patterns like
consistent product configurations (colors, brands) across positions.

Paper config: 60 epochs, batch_size=8, lr=5e-06, dropout=0.1
"""

import torch
import torch.nn as nn
from transformers import BertModel


class SequentialSentenceClassifier(nn.Module):
    """
    BERT + BiLSTM sequential sentence classifier.

    Step 1: Each sentence is encoded independently by BERT (shared weights)
    Step 2: Sentence embeddings are processed by a BiLSTM to capture context
    Step 3: Each sentence is classified using its context-enriched representation

    This architecture allows the model to leverage information from
    surrounding sentences when classifying each position.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = "bert-base-uncased",
        dropout: float = 0.1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # BiLSTM for sequential context
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)  # *2 for bidirectional
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, labels=None, num_sentences=None):
        """
        Args:
            input_ids: (batch_size, max_num_sentences, seq_len)
            attention_mask: (batch_size, max_num_sentences, seq_len)
            labels: (batch_size, max_num_sentences) optional, -100 for padding
            num_sentences: (batch_size,) actual number of sentences per sample

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size, max_num_sent, seq_len = input_ids.shape

        # Reshape to process all sentences
        # (batch * max_num_sent, seq_len)
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        # Process through BERT in smaller chunks to avoid OOM on CPU
        chunk_size = 16
        sentence_embeddings_list = []

        for i in range(0, flat_input_ids.shape[0], chunk_size):
            chunk_input_ids = flat_input_ids[i : i + chunk_size]
            chunk_attention_mask = flat_attention_mask[i : i + chunk_size]

            chunk_outputs = self.bert(
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
            )
            # Get [CLS] token embeddings for each sentence
            sentence_embeddings_list.append(chunk_outputs.last_hidden_state[:, 0, :])

        # Combine chunks and reshape
        # (batch * max_num_sent, hidden_size) → (batch, max_num_sent, hidden_size)
        sentence_embeddings = torch.cat(sentence_embeddings_list, dim=0)
        sentence_embeddings = sentence_embeddings.view(
            batch_size, max_num_sent, self.hidden_size
        )

        # Process sequence of sentence embeddings with BiLSTM
        # This is where context from surrounding sentences is captured
        lstm_output, _ = self.lstm(sentence_embeddings)  # (batch, max_sent, lstm_hidden*2)

        lstm_output = self.dropout(lstm_output)
        logits = self.classifier(lstm_output)  # (batch, max_sent, num_labels)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Reshape for loss computation
            logits_flat = logits.view(-1, self.num_labels)
            labels_flat = labels.view(-1)
            result["loss"] = loss_fn(logits_flat, labels_flat)

        return result

    def get_num_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
