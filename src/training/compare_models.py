"""
Model comparison and analysis script.

Compares SinSent vs SeqSent models and generates:
- Comparison tables
- Visualization plots (F1, confusion matrices, confidence distributions)
- Statistical analysis
- Latency benchmarking

This is the core of the Track-I analysis-based research.
"""

import sys
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.preprocess import (  # noqa: E402
    create_dataloaders,
    NUM_LABELS,
    ID_TO_LABEL,
)
from src.models.single_sentence import SingleSentenceClassifier  # noqa: E402
from src.models.sequential_sentence import SequentialSentenceClassifier  # noqa: E402
from src.models.utils import (  # noqa: E402
    get_confusion_matrix,
    count_parameters,
    get_device,
)
from src.training.train import evaluate_sinsent, evaluate_seqsent  # noqa: E402


def load_model(model_type, device):
    """Load a trained model from checkpoint."""
    checkpoint_path = f"models/{model_type}_best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: {checkpoint_path} not found!")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # nosec B614
    saved_args = checkpoint["args"]

    if model_type == "sinsent":
        model = SingleSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name=saved_args.get("model_name", "bert-base-uncased"),
            dropout=saved_args.get("dropout", 0.3),
        )
    else:
        model = SequentialSentenceClassifier(
            num_labels=NUM_LABELS,
            model_name=saved_args.get("model_name", "bert-base-uncased"),
            dropout=saved_args.get("dropout", 0.1),
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, saved_args


def benchmark_latency(model, dataloader, device, model_type, n_runs=3):
    """Benchmark inference latency."""
    model.eval()
    times = []

    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            for batch in dataloader:
                if model_type == "sinsent":
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    _ = model(input_ids, attention_mask)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    num_sentences = batch["num_sentences"]
                    _ = model(input_ids, attention_mask, num_sentences=num_sentences)
        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def plot_metric_comparison(sinsent_metrics, seqsent_metrics, output_dir):
    """Create bar chart comparing metrics between models."""
    metrics_to_plot = [
        ("f1_micro", "F1 (Micro)"),
        ("f1_macro", "F1 (Macro)"),
        ("precision_micro", "Precision (Micro)"),
        ("recall_micro", "Recall (Micro)"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    sinsent_vals = [sinsent_metrics.get(m[0], 0) for m in metrics_to_plot]
    seqsent_vals = [seqsent_metrics.get(m[0], 0) for m in metrics_to_plot]

    bars1 = ax.bar(
        x - width / 2,
        sinsent_vals,
        width,
        label="SinSent",
        color="#4C72B0",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        seqsent_vals,
        width,
        label="SeqSent",
        color="#DD8452",
        edgecolor="white",
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: SinSent vs SeqSent", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics_to_plot], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "metric_comparison.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved metric_comparison.png")


def plot_confusion_matrices(
    sinsent_preds, sinsent_labels, seqsent_preds, seqsent_labels, output_dir
):
    """Plot side-by-side confusion matrices."""
    label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, preds, labels, title in [
        (axes[0], sinsent_preds, sinsent_labels, "SinSent"),
        (axes[1], seqsent_preds, seqsent_labels, "SeqSent"),
    ]:
        cm = get_confusion_matrix(preds, labels)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=label_names,
            yticklabels=label_names,
            ax=ax,
        )
        ax.set_title(f"{title} Confusion Matrix", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved confusion_matrices.png")


def plot_confidence_distribution(sinsent_probs, seqsent_probs, output_dir):
    """Plot prediction confidence distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, probs, title in [
        (axes[0], sinsent_probs, "SinSent"),
        (axes[1], seqsent_probs, "SeqSent"),
    ]:
        max_probs = probs.max(axis=1)
        ax.hist(max_probs, bins=50, color="#4C72B0", alpha=0.7, edgecolor="white")
        ax.axvline(
            x=np.median(max_probs),
            color="red",
            linestyle="--",
            label=f"Median: {np.median(max_probs):.3f}",
        )
        ax.set_title(f"{title} Prediction Confidence", fontsize=13, fontweight="bold")
        ax.set_xlabel("Max Prediction Probability")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "confidence_distribution.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("  Saved confidence_distribution.png")


def plot_per_class_f1(
    sinsent_preds, sinsent_labels, seqsent_preds, seqsent_labels, output_dir
):
    """Plot per-class F1 comparison."""
    from sklearn.metrics import f1_score

    label_names = [ID_TO_LABEL[i] for i in range(NUM_LABELS)]

    sinsent_f1 = f1_score(sinsent_labels, sinsent_preds, average=None)
    seqsent_f1 = f1_score(seqsent_labels, seqsent_preds, average=None)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(label_names))
    width = 0.35

    ax.bar(x - width / 2, sinsent_f1, width, label="SinSent", color="#4C72B0")
    ax.bar(x + width / 2, seqsent_f1, width, label="SeqSent", color="#DD8452")

    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "per_class_f1.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved per_class_f1.png")


def compare_models(args):
    """Run full model comparison analysis."""
    device = get_device()
    output_dir = "results/comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MODEL COMPARISON: SinSent vs SeqSent")
    print("=" * 60)

    # Load models
    sinsent_model, sinsent_args = load_model("sinsent", device)
    seqsent_model, seqsent_args = load_model("seqsent", device)

    if sinsent_model is None or seqsent_model is None:
        print("ERROR: Both models must be trained first!")
        return

    # Create dataloaders
    print("\nLoading test data...")
    _, _, sinsent_test_loader = create_dataloaders(
        data_dir=args.data_dir,
        model_type="sinsent",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    _, _, seqsent_test_loader = create_dataloaders(
        data_dir=args.data_dir,
        model_type="seqsent",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Evaluate both models
    print("\nEvaluating SinSent...")
    _, sinsent_metrics, sinsent_preds, sinsent_labels, sinsent_probs = evaluate_sinsent(
        sinsent_model, sinsent_test_loader, device
    )

    print("Evaluating SeqSent...")
    _, seqsent_metrics, seqsent_preds, seqsent_labels, seqsent_probs = evaluate_seqsent(
        seqsent_model, seqsent_test_loader, device
    )

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'SinSent':>10} {'SeqSent':>10} {'Δ':>10}")
    print("-" * 55)
    for metric in [
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "precision_micro",
        "recall_micro",
    ]:
        s = sinsent_metrics.get(metric, 0)
        q = seqsent_metrics.get(metric, 0)
        delta = q - s
        print(f"  {metric:<23} {s:>10.4f} {q:>10.4f} {delta:>+10.4f}")

    # Parameter comparison
    print(f"\n{'='*60}")
    print("MODEL PARAMETERS")
    print(f"{'='*60}")
    sin_params = count_parameters(sinsent_model)
    seq_params = count_parameters(seqsent_model)
    print(
        f"  SinSent: {sin_params['total_parameters']:,} total, {sin_params['trainable_parameters']:,} trainable"
    )
    print(
        f"  SeqSent: {seq_params['total_parameters']:,} total, {seq_params['trainable_parameters']:,} trainable"
    )

    # Latency benchmarking
    print(f"\n{'='*60}")
    print("LATENCY BENCHMARKING")
    print(f"{'='*60}")
    sinsent_latency = benchmark_latency(
        sinsent_model, sinsent_test_loader, device, "sinsent"
    )
    seqsent_latency = benchmark_latency(
        seqsent_model, seqsent_test_loader, device, "seqsent"
    )
    print(
        f"  SinSent: {sinsent_latency['mean_time']:.2f}s ± {sinsent_latency['std_time']:.2f}s"
    )
    print(
        f"  SeqSent: {seqsent_latency['mean_time']:.2f}s ± {seqsent_latency['std_time']:.2f}s"
    )

    # Generate plots
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    plot_metric_comparison(sinsent_metrics, seqsent_metrics, output_dir)
    plot_confusion_matrices(
        sinsent_preds, sinsent_labels, seqsent_preds, seqsent_labels, output_dir
    )
    plot_confidence_distribution(sinsent_probs, seqsent_probs, output_dir)
    plot_per_class_f1(
        sinsent_preds, sinsent_labels, seqsent_preds, seqsent_labels, output_dir
    )

    # Save full comparison results
    comparison = {
        "sinsent": {
            "metrics": sinsent_metrics,
            "parameters": sin_params,
            "latency": sinsent_latency,
        },
        "seqsent": {
            "metrics": seqsent_metrics,
            "parameters": seq_params,
            "latency": seqsent_latency,
        },
    }

    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n✓ Comparison complete! Results saved to {output_dir}/")
    return comparison


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare SinSent vs SeqSent models")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=None)

    args = parser.parse_args()
    compare_models(args)
