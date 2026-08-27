#!/usr/bin/env python3
"""
DeepSRGM -- train the original LSTM-based DeepSRGM on the selected 20 ragas.

This is the training entry point for the 20-raga experiment.  It reuses the
existing model implementation in ``src/deepSRGM.py`` (LSTM, hidden=768,
attention, fc1 768->384, fc2 384->num_classes) and the original training
methodology (Adam lr=1e-4, CrossEntropyLoss, batch size 40, random
subsequence sampling).  No architectural change is made.

The held-out set (built by ``deepsrgm_data.build_dataset`` at the recording
level) is used as the validation set during training so the original
train/test two-way protocol is preserved; validation metrics and best-checkpoint
selection use it.

Usage:
    python3 ragasense/train_deepsrgm.py --epochs 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Make the existing DeepSRGM model importable.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from deepSRGM import DeepSRGM            # noqa: E402  (existing model)
import deepsrgm_data as data             # noqa: E402  (this package)


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepSRGM on 20 ragas")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--windows", type=int, default=data.N_WINDOWS)
    p.add_argument("--seq-len", type=int, default=data.SEQ_LEN)
    p.add_argument("--seed", type=int, default=data.SEED)
    p.add_argument("--margin-vote", type=float, default=0.6,
                   help="majority-voting threshold (original=0.6)")
    p.add_argument("--results-dir", type=str,
                   default=str(PROJECT_ROOT / "ragasense" / "results" / "deepsrgm"))
    return p.parse_args()


def recording_accuracy(model, X, Y, n_windows, threshold, device):
    """Recording-level accuracy using majority voting (original procedure)."""
    model.eval()
    n_recs = len(Y) // n_windows
    correct = 0
    with torch.no_grad():
        for r in range(n_recs):
            start = r * n_windows
            block = X[start:start + n_windows].to(device)
            out = model(block)
            preds = torch.argmax(out, axis=-1)
            labels = Y[start:start + n_windows].to(device)
            matched = float(torch.sum(preds == labels))
            if matched / len(preds) >= threshold:
                correct += 1
    return correct / n_recs if n_recs else 0.0


def window_metrics(model, X, Y, device):
    """Return (window accuracy, mean loss) over all windows (naive eval)."""
    model.eval()
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    n = len(Y)
    with torch.no_grad():
        for i in range(0, n, 128):
            xb = X[i:i + 128].to(device)
            yb = Y[i:i + 128].to(device)
            out = model(xb)
            loss_sum += float(criterion(out, yb)) * len(yb)
            preds = torch.argmax(out, axis=-1)
            correct += int((preds == yb).sum().item())
    return correct / n if n else 0.0, (loss_sum / n if n else 0.0)

def save_plots(history, results_dir):
    ep = [h["epoch"] for h in history]
    tl = [h["train_loss"] for h in history]
    vl = [h["val_loss"] for h in history]
    va = [h["val_recording_accuracy"] for h in history]

    plt.figure()
    plt.plot(ep, tl, label="train_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Training Loss"); plt.grid(alpha=0.3)
    plt.savefig(results_dir / "training_loss.png", dpi=120); plt.close()

    plt.figure()
    plt.plot(ep, vl, label="val_loss", color="tab:red")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Validation Loss"); plt.grid(alpha=0.3)
    plt.savefig(results_dir / "validation_loss.png", dpi=120); plt.close()

    plt.figure()
    plt.plot(ep, tl, label="train_loss")
    plt.plot(ep, vl, label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Training vs Validation Loss"); plt.grid(alpha=0.3)
    plt.savefig(results_dir / "training_vs_validation_loss.png", dpi=120); plt.close()

    plt.figure()
    plt.plot(ep, va, label="val_recording_accuracy", color="tab:green")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.ylim(0, 1)
    plt.legend(); plt.title("Validation Accuracy"); plt.grid(alpha=0.3)
    plt.savefig(results_dir / "validation_accuracy.png", dpi=120); plt.close()


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_model_config(ds):
    return {
        "rnn": "lstm", "input_length": ds["seq_len"], "embedding_size": 128,
        "hidden_size": 768, "num_layers": 1, "num_classes": ds["n_classes"],
        "vocab_size": ds["vocab_size"], "drop_prob": 0.3,
    }

def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Building dataset (recording-level split)...")
    ds = data.build_dataset(
        n_windows=args.windows, length=args.seq_len, seed=args.seed
    )
    X_train = torch.from_numpy(ds["X_train"]).long()
    Y_train = torch.from_numpy(ds["Y_train"]).long()
    X_test = torch.from_numpy(ds["X_test"]).long()
    Y_test = torch.from_numpy(ds["Y_test"]).long()
    print(f"  Train: X{X_train.shape} Y{Y_train.shape}")
    print(f"  Held-out (validation): X{X_test.shape} Y{Y_test.shape}")

    # Persist held-out data + split records so evaluation reuses the SAME split.
    np.save(results_dir / "X_test.npy", ds["X_test"])
    np.save(results_dir / "Y_test.npy", ds["Y_test"])
    import csv
    with open(results_dir / "split_info.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ds["split_info"][0].keys()))
        writer.writeheader()
        writer.writerows(ds["split_info"])

    print("\n[2/4] Building model (original DeepSRGM, LSTM)...")
    model_cfg = build_model_config(ds)
    model = DeepSRGM(**model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    trainset = TensorDataset(X_train, Y_train)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size)

    print(f"  classes={ds['n_classes']} vocab={ds['vocab_size']} "
          f"windows/recording={args.windows}")

    print("\n[3/4] Training...")
    history = []
    best_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_correct += int(
                (torch.argmax(outputs, axis=-1) == labels.to(device)).sum().item()
            )
            train_total += labels.size(0)

            if (i + 1) % 15 == 0:
                print(f"  Epoch {epoch}/{args.epochs} | Batch {i+1}/{len(trainloader)} | "
                      f"Loss: {(running_loss/15):.3f}")
                epoch_loss += running_loss
                running_loss = 0.0

        if epoch_loss > 0.0:
            train_loss = epoch_loss / (len(trainloader) // 15 * 15)
        else:
            train_loss = running_loss / len(trainloader)
        train_loss = float(train_loss)
        train_acc = train_correct / train_total if train_total else 0.0

        val_loss, val_win_acc = window_metrics(model, X_test, Y_test, device)
        val_rec_acc = recording_accuracy(
            model, X_test, Y_test, args.windows, args.margin_vote, device
        )

        if val_rec_acc > best_acc:
            best_acc = val_rec_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_window_accuracy": val_win_acc,
            "val_recording_accuracy": val_rec_acc,
            "learning_rate": args.lr,
            "epoch_time": time.time() - t0,
        })
        print(f"  Epoch {epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.3f} "
              f"| val_loss {val_loss:.4f} | val_win_acc {val_win_acc:.3f} "
              f"| val_rec_acc {val_rec_acc:.3f}")

    print("\n[4/4] Saving checkpoints and artifacts...")
    best_checkpoint = {
        "model_state_dict": best_state,
        "model_config": model_cfg,
        "epoch": int(max(h["epoch"] for h in history)),
        "best_val_recording_accuracy": best_acc,
        "num_classes": ds["n_classes"],
        "class_mapping": ds["class_mapping"],
        "preprocessing": {"k": ds["k"], "seq_len": ds["seq_len"],
                          "n_windows": ds["n_windows"], "seed": args.seed},
    }
    final_checkpoint = dict(best_checkpoint)
    final_checkpoint["model_state_dict"] = model.state_dict()
    final_checkpoint["epoch"] = args.epochs

    torch.save(best_checkpoint, results_dir / "best_model.pt")
    torch.save(final_checkpoint, results_dir / "final_model.pt")

    import csv
    with open(results_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    config = {
        "experiment": "deepSRGM_20_ragas",
        "num_classes": ds["n_classes"],
        "model": model_cfg,
        "preprocessing": {"k": ds["k"], "seq_len": ds["seq_len"],
                          "n_windows": ds["n_windows"], "seed": args.seed},
        "training": {"epochs": args.epochs, "lr": args.lr,
                     "batch_size": args.batch_size,
                     "margin_vote_threshold": args.margin_vote},
        "dataset": {"root": str(data.DATASET_ROOT),
                    "train_per_raga": data.TRAIN_PER_RAGA},
    }
    save_json(results_dir / "config.json", config)
    save_json(results_dir / "class_mapping.json", ds["class_mapping"])

    save_plots(history, results_dir)
    print("Done. Artifacts in:", results_dir)


if __name__ == "__main__":
    main()


