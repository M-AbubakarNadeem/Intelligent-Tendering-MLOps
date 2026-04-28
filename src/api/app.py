"""
Flask REST API for sentence classification with Prometheus metrics.

Provides endpoints for:
- /predict: Classify sentences using the trained model
- /health: Health check
- /metrics: Prometheus metrics endpoint

This corresponds to the paper's Flask-based container serving the ML solution.
"""

import os
import sys
import json
import time
import torch
import logging
from flask import Flask, request, jsonify
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from transformers import BertTokenizer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.single_sentence import SingleSentenceClassifier
from src.models.sequential_sentence import SequentialSentenceClassifier
from src.data.preprocess import NUM_LABELS, ID_TO_LABEL

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# ─── Prometheus Metrics ──────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["model_type", "status"],
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_type"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Prediction confidence score",
    ["model_type"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
)
MODEL_LOADED = Gauge("model_loaded", "Whether a model is loaded", ["model_type"])
ACTIVE_REQUESTS = Gauge("active_requests", "Number of active requests")

# ─── Global Model State ─────────────────────────────────────────────
model = None
tokenizer = None
model_type = None
device = None


def load_model_from_checkpoint():
    """Load the best available model on startup."""
    global model, tokenizer, model_type, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_type = os.environ.get("MODEL_TYPE", "seqsent")
    model_name = os.environ.get("MODEL_NAME", "bert-base-uncased")
    checkpoint_path = f"models/{target_type}_best.pt"

    if not os.path.exists(checkpoint_path):
        logger.warning(f"No checkpoint found at {checkpoint_path}")
        MODEL_LOADED.labels(model_type=target_type).set(0)
        return False

    logger.info(f"Loading {target_type} model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]

    if target_type == "sinsent":
        model = SingleSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name=saved_args.get("model_name", model_name),
            dropout=saved_args.get("dropout", 0.3),
        )
    else:
        model = SequentialSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name=saved_args.get("model_name", model_name),
            dropout=saved_args.get("dropout", 0.1),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    model_type = target_type

    tokenizer = BertTokenizer.from_pretrained(
        saved_args.get("model_name", model_name)
    )

    MODEL_LOADED.labels(model_type=model_type).set(1)
    logger.info(f"Model loaded successfully! Type: {model_type}, Device: {device}")
    return True


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": str(device),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict sentence labels.

    For SinSent: expects {"sentences": ["sentence1", "sentence2", ...]}
    For SeqSent: expects {"sentences": ["sentence1", "sentence2", ...]}
                 (processes them as a sequence)
    """
    if model is None:
        REQUEST_COUNT.labels(model_type="none", status="error").inc()
        return jsonify({"error": "No model loaded"}), 503

    ACTIVE_REQUESTS.inc()
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or "sentences" not in data:
            REQUEST_COUNT.labels(model_type=model_type, status="error").inc()
            return jsonify({"error": "Missing 'sentences' field"}), 400

        sentences = data["sentences"]
        max_len = int(os.environ.get("MAX_LEN", 128))

        with torch.no_grad():
            if model_type == "sinsent":
                results = _predict_sinsent(sentences, max_len)
            else:
                results = _predict_seqsent(sentences, max_len)

        latency = time.time() - start_time
        REQUEST_COUNT.labels(model_type=model_type, status="success").inc()
        REQUEST_LATENCY.labels(model_type=model_type).observe(latency)

        for r in results:
            PREDICTION_CONFIDENCE.labels(model_type=model_type).observe(
                r["confidence"]
            )

        ACTIVE_REQUESTS.dec()

        return jsonify({
            "model_type": model_type,
            "predictions": results,
            "latency_ms": round(latency * 1000, 2),
        })

    except Exception as e:
        ACTIVE_REQUESTS.dec()
        REQUEST_COUNT.labels(model_type=model_type, status="error").inc()
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500


def _predict_sinsent(sentences, max_len):
    """Run SinSent prediction on individual sentences."""
    results = []
    for sent in sentences:
        encoding = tokenizer(
            sent,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs["logits"], dim=-1).cpu().numpy()[0]
        pred_class = int(probs.argmax())

        results.append({
            "sentence": sent,
            "predicted_label": ID_TO_LABEL[pred_class],
            "confidence": round(float(probs[pred_class]), 4),
            "all_probabilities": {
                ID_TO_LABEL[i]: round(float(p), 4) for i, p in enumerate(probs)
            },
        })
    return results


def _predict_seqsent(sentences, max_len):
    """Run SeqSent prediction on a sequence of sentences."""
    # Tokenize all sentences
    input_ids_list = []
    attention_mask_list = []

    for sent in sentences:
        encoding = tokenizer(
            sent,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(encoding["input_ids"])
        attention_mask_list.append(encoding["attention_mask"])

    # Stack into batch with sequence dimension
    input_ids = torch.stack([t.squeeze(0) for t in input_ids_list]).unsqueeze(0).to(device)
    attention_mask = torch.stack([t.squeeze(0) for t in attention_mask_list]).unsqueeze(0).to(device)
    num_sentences = torch.tensor([len(sentences)])

    outputs = model(input_ids, attention_mask, num_sentences=num_sentences)
    logits = outputs["logits"][0, : len(sentences), :]  # (num_sent, num_labels)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for i, sent in enumerate(sentences):
        pred_class = int(probs[i].argmax())
        results.append({
            "sentence": sent,
            "predicted_label": ID_TO_LABEL[pred_class],
            "confidence": round(float(probs[i][pred_class]), 4),
            "all_probabilities": {
                ID_TO_LABEL[j]: round(float(p), 4) for j, p in enumerate(probs[i])
            },
        })
    return results


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/", methods=["GET"])
def index():
    """Root endpoint with API info."""
    return jsonify({
        "name": "Intelligent Tendering Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Classify sentences",
            "/health": "GET - Health check",
            "/metrics": "GET - Prometheus metrics",
        },
    })


# Load model on startup
with app.app_context():
    load_model_from_checkpoint()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
