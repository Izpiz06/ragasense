#!/usr/bin/env python3
"""
DeepSRGM -- evaluate the trained LSTM-based DeepSRGM on the held-out set.

Loads the saved checkpoint + the held-out data (persisted by the training
script so the exact same recording-level split is reused), runs inference,
aggregates window predictions into a single recording-level prediction via
majority voting (original procedure), and reports:

    accuracy, macro/weighted F1, macro precision/recall, top-3 accuracy

Outputs in the results directory:
    recording_predictions.csv
    classification_report.csv
    per_raga_results.csv
    confusion_matrix.csv
    confusion_matrix.png
    normalized_confusion_matrix.png

Usage:
    python3 ragasense/evaluate_deepsrgm.py --checkpoint best_model.pt
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepSRGM import DeepSRGM            # noqa: E402  (existing model)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepSRGM on 20 ragas")
    p.add_argument("--results-dir", type=str,
                   default=str(PROJECT_ROOT / "ragasense" / "results" / "deepsrgm"))
    p.add_argument("--checkpoint", type=str, default="best_model.pt")
    p.add_argument("--margin-vote", type=float, default=0.6)
    return p.parse_args()


def load(result_dir, name, as_text=False):
    path = result_dir / name
    if as_text:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return np.load(path)


def recording_predictions(model, X, Y, n_windows, device, class_mapping):
    """Majority-vote each recording into a single class (original procedure)."""
    model.eval()
    n_recs = len(Y) // n_windows
    rows = []
    with torch.no_grad():
        for r in range(n_recs):
            start = r * n_windows
            block = X[start:start + n_windows].to(device)
            out = model(block)
            win_preds = torch.argmax(out, axis=-1)
            true = int(Y[start].item())
            votes = torch.bincount(win_preds, minlength=len(class_mapping))
            pred = int(votes.argmax().item())
            frac = float(votes[pred]) / n_windows
            rows.append({
                "recording_idx": r,
                "true_class": true,
                "true_raga": class_mapping[str(true)],
                "pred_class": pred,
                "pred_raga": class_mapping[str(pred)],
                "votes": frac,
                "correct": pred == true,
            })
    return rows


def top3_window_accuracy(model, X, Y, device):
    """Fraction of windows whose true class is among the top-3 outputs."""
    model.eval()
    n = len(Y)
    hit = 0
    with torch.no_grad():
        for i in range(0, n, 128):
            xb = X[i:i + 128].to(device)
            yb = Y[i:i + 128].to(device)
            out = model(xb)
            _, top3 = torch.topk(out, k=3, dim=-1)
            hit += int((top3 == yb.unsqueeze(1)).any(dim=1).sum().item())
    return hit / n if n else 0.0


def compute_class_metrics(rows, n_classes):
    """Manually compute precision/recall/f1 per class from recording preds."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for row in rows:
        cm[row["true_class"], row["pred_class"]] += 1
    accuracy = float(np.trace(cm)) / max(len(rows), 1)

    per_class = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        support = int(cm[c, :].sum())
        pred_count = int(cm[:, c].sum())
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_class.append({
            "class": c, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        })

    considered = [pc for pc in per_class if pc["support"] > 0]
    macro = {m: float(np.mean([pc[m] for pc in considered])) for m in
             ("precision", "recall", "f1")}
    total = sum(pc["support"] for pc in per_class) or 1.0
    weighted = {m: float(sum(pc[m] * pc["support"] for pc in per_class) / total)
                for m in ("precision", "recall", "f1")}
    return accuracy, cm, per_class, macro, weighted

def plot_confusion(cm, class_mapping, result_dir, normalized=False):
    if normalized:
        sums = cm.sum(axis=1, keepdims=True)
        display = np.divide(cm, sums, out=np.zeros_like(cm, dtype=float),
                            where=sums != 0)
        fname = "normalized_confusion_matrix.png"
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        display = cm
        fname = "confusion_matrix.png"
        fmt = "d"
        title = "Confusion Matrix"

    labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
    plt.figure(figsize=(11, 9))
    plt.imshow(display, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    thresh = display.max() / 2 if display.max() else 0.5
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = display[i, j]
            if normalized or val > 0:
                plt.text(j, i, format(val, fmt),
                         ha="center", va="center",
                         color="white" if val > thresh else "black")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(title); plt.tight_layout()
    plt.savefig(result_dir / fname, dpi=120)
    plt.close()


def write_csv(result_dir, name, fieldnames, rows):
    with open(result_dir / name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    result_dir = Path(args.results_dir)

    config = load(result_dir, "config.json", as_text=True)
    class_mapping = load(result_dir, "class_mapping.json", as_text=True)
    class_mapping = {str(k): v for k, v in class_mapping.items()}
    n_classes = len(class_mapping)
    n_windows = config["preprocessing"]["n_windows"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    checkpoint = torch.load(result_dir / args.checkpoint, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = DeepSRGM(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    X = torch.from_numpy(load(result_dir, "X_test.npy")).long()
    Y = torch.from_numpy(load(result_dir, "Y_test.npy")).long()
    print(f"Held-out: X{X.shape} Y{Y.shape} ({X.shape[0] // n_windows} recordings)")

    rows = recording_predictions(model, X, Y, n_windows, device, class_mapping)
    accuracy, cm, per_class, macro, weighted = compute_class_metrics(rows, n_classes)
    top3 = top3_window_accuracy(model, X, Y, device)

    print("=" * 60)
    print(f"Recording accuracy : {accuracy * 100:.2f}%")
    print(f"Macro F1           : {macro['f1']:.4f}")
    print(f"Weighted F1        : {weighted['f1']:.4f}")
    print(f"Macro precision    : {macro['precision']:.4f}")
    print(f"Macro recall       : {macro['recall']:.4f}")
    print(f"Top-3 (window) acc : {top3 * 100:.2f}%")
    print("=" * 60)

    write_csv(result_dir, "recording_predictions.csv",
              ["recording_idx", "true_class", "true_raga", "pred_class",
               "pred_raga", "votes", "correct"],
              [{**r, "correct": int(r["correct"])} for r in rows])

    report_rows = [
        {**pc, "raga": class_mapping[str(pc["class"])]}
        for pc in per_class
    ]
    for kind, vals in (("macro avg", macro), ("weighted avg", weighted)):
        report_rows.append({"class": kind, "raga": "", "precision": vals["precision"],
                            "recall": vals["recall"], "f1": vals["f1"],
                            "support": len(rows)})
    write_csv(result_dir, "classification_report.csv",
              ["class", "raga", "precision", "recall", "f1", "support"],
              report_rows)

    write_csv(result_dir, "per_raga_results.csv",
              ["class", "raga", "precision", "recall", "f1", "support"],
              report_rows)

    with open(result_dir / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + [class_mapping[str(i)] for i in range(n_classes)])
        for i in range(n_classes):
            writer.writerow([class_mapping[str(i)]] + list(cm[i]))

    plot_confusion(cm, class_mapping, result_dir, normalized=False)
    plot_confusion(cm, class_mapping, result_dir, normalized=True)

    print("Evaluation artifacts saved in:", result_dir)


if __name__ == "__main__":
    main()

