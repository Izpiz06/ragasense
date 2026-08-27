#!/usr/bin/env python3
"""
DeepSRGM -- 20-raga dataset loader & preprocessing.

This module adapts the ORIGINAL DeepSRGM preprocessing to the selected
20-raga RagaDataset subset.  It reuses the exact original pipeline:

    feature = round( 1200 * log2(pitch / tonic) * (k/100) ).clip(0)   (k = 5)

i.e. tonic normalisation with the recording tonic (fn = 1200*log2(f/T))
and quantization to 5 frequency levels per semitone.  For every recording
we draw N random temporal subsequences of length `seq_len` (5000), exactly
like the original `get_feature()`.

The 20-raga selection and the Unicode-aware name normalisation are imported
from ``ragasense/deep_analyze_20_ragas.py`` (single source of truth), so this
file never hard-codes the raga list.

SPLIT (critical, recording-level)
---------------------------------
The split is performed at the RECORDING level *before* any subsequence is
sampled, so no window from one recording can appear in two splits.  This
matches the original methodology, which effectively used the first 9
recordings of each raga for training and the remaining for the held-out set,
but here the assignment is randomized and deterministic (seeded).

  Carnatic   : 12 recordings/raga  -> 9 train / 3 held-out
  Hindustani : 10 recordings/raga  -> 9 train / 1 held-out

Only what is required is computed; no dataset file is modified.
"""

from pathlib import Path
import json

import numpy as np

# Single source of truth for the selected ragas + Unicode normalisation.
from deep_analyze_20_ragas import SELECTED_RAGAS, normalize_text

DATASET_ROOT = Path(
    "/home/izpiz/coding/automatic-raga-recognition/dataset/RagaDataset"
)

# Original preprocessing constants.
K = 5                      # frequency levels per semitone (k/100 * 1200 cents)
SEQ_LEN = 5000             # subsequence length (input_length to the model)
N_WINDOWS = 200            # random subsequences per recording
SEED = 42                  # default random seed for reproducibility

# Original embedding vocabulary was 209 tokens (0..208).  Our 20 selected
# ragas span a slightly wider pitch range (max token ~216), so we enlarge
# the vocabulary to 256 -- a pure compatibility fix that keeps the original
# quantization formula intact (no silent clipping of high pitches).
VOCAB_SIZE = 256

# Recordings per raga used for the recording-level split.
TRAIN_PER_RAGA = {
    "Carnatic": 9,
    "Hindustani": 9,
}


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_raga_name_mapping(tradition):
    """Return {raga_id: raga_name} from the dataset _info_ mapping file."""
    path = (
        DATASET_ROOT
        / tradition
        / "_info_"
        / "ragaId_to_ragaName_mapping.json"
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_raga_ids(tradition, names):
    """Map selected raga names -> dataset raga ids that have features.

    Matching is Unicode-aware (normalize_text), but some names collide under
    normalisation (e.g. Carnatic "Tōḍi" vs Hindustani "Tōḍī" both -> "todi").
    To avoid silently pairing a raga with the wrong id, we:
      1. consider all ids whose normalised name matches the target,
      2. prefer an exact (diacritic-preserving) name match,
      3. only keep ids that actually have a feature directory in this
         tradition.

    Returns a list of raga ids (usually a single element).
    """
    mapping = load_raga_name_mapping(tradition)
    feature_root = DATASET_ROOT / tradition / "features"
    target_norm = normalize_text(names[0])
    exact_matches = []
    norm_matches = []
    for rid, nm in mapping.items():
        if normalize_text(nm) != target_norm:
            continue
        if (feature_root / rid).exists():
            if nm == names[0]:
                exact_matches.append(rid)
            norm_matches.append(rid)

    candidates = exact_matches if exact_matches else norm_matches
    if not candidates:
        raise ValueError(
            f"Could not resolve a feature-bearing raga id for "
            f"{tradition}/{names[0]} (normalised '{target_norm}')."
        )
    return candidates

# ---------------------------------------------------------------------------
# Feature reading (faithful to the original get_feature)
# ---------------------------------------------------------------------------

def read_pitch_file(pitch_path):
    """Read a .pitch file returning the pitch column (last tab field).

    The original code did `line.split("\t")[-1]` and `eval(...)`.
    Unparseable / empty lines are treated as unvoiced (0.0) -- equivalent to
    the original behaviour where unvoiced frames are already 0.0.
    """
    values = []
    with open(pitch_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                values.append(float(line.split("\t")[-1]))
            except (ValueError, IndexError):
                values.append(0.0)
    return np.asarray(values, dtype=np.float64)


def read_tonic(base_pitch_path):
    """Read the recording tonic (prefer .tonicFine, fall back to .tonic)."""
    lookup = str(base_pitch_path)[: -len(".pitch")]
    for suffix in (".tonicFine", ".tonic"):
        path = Path(lookup + suffix)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return float(f.read().strip())
    raise FileNotFoundError(
        f"No tonic file (.tonicFine/.tonic) for {base_pitch_path}"
    )


def quantize_feature(pitch, tonic):
    """Tonic-normalise and quantize a pitch track (original formula)."""
    # np.round(...) returns floats; we cast to int32 for the token table.
    with np.errstate(divide="ignore", invalid="ignore"):
        feature = np.round(
            1200.0 * np.log2(pitch / tonic) * (K / 100.0)
        ).clip(0)
    return np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.int32
    )


def sample_windows(feature, n=N_WINDOWS, length=SEQ_LEN, rng=None):
    """Draw `n` random temporal subsequences of `length` (original sampling)."""
    if rng is None:
        rng = np.random
    if feature.size <= length:
        raise ValueError(
            f"Feature length {feature.size} <= seq_len {length}; "
            f"cannot sample {length}-long windows."
        )
    windows = np.empty((n, length), dtype=np.int32)
    for i in range(n):
        c = rng.randint(0, feature.size - length)
        windows[i] = feature[c:c + length]
    return windows


# ---------------------------------------------------------------------------
# Recording discovery
# ---------------------------------------------------------------------------

def discover_recordings(tradition, raga_id):
    """Return a list of .pitch file Paths for a raga (each == one recording)."""
    feature_dir = DATASET_ROOT / tradition / "features" / raga_id
    if not feature_dir.exists():
        return []
    return sorted(feature_dir.rglob("*.pitch"))



# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    n_windows=N_WINDOWS,
    length=SEQ_LEN,
    train_per_raga=None,
    seed=SEED,
    verbose=True,
):
    """Build train / held-out subsequence arrays at the recording level.

    Returns a dict with:
        X_train, Y_train, X_test, Y_test   (numpy arrays)
        class_mapping (dict idx -> raga name, deterministic order)
        n_classes, seq_len, n_windows, vocab_size, k
        split_info   (list of per-recording metadata for audit/leak checks)
    """
    if train_per_raga is None:
        train_per_raga = TRAIN_PER_RAGA

    rng = np.random.RandomState(seed)

    # Deterministic class ordering: Carnatic 0..9, Hindustani 10..19.
    class_order = []
    for tradition in ("Carnatic", "Hindustani"):
        for name in SELECTED_RAGAS[tradition]:
            class_order.append((tradition, name))
    class_mapping = {i: name for i, (_, name) in enumerate(class_order)}

    X_train, Y_train = [], []
    X_test, Y_test = [], []
    split_info = []

    for class_idx, (tradition, name) in enumerate(class_order):
        raga_ids = resolve_raga_ids(tradition, [name])
        recordings = []
        for raga_id in raga_ids:
            recordings.extend(discover_recordings(tradition, raga_id))
        raga_id = ",".join(raga_ids)

        if not recordings:
            raise ValueError(f"No recordings for {tradition}/{name} ({raga_id})")

        # --- recording-level split (before sampling -> no leakage) ----------
        recordings = list(recordings)
        rng.shuffle(recordings)

        n_train = min(train_per_raga.get(tradition, 9), len(recordings) - 1)
        train_recs = recordings[:n_train]
        test_recs = recordings[n_train:]

        if not test_recs:
            raise ValueError(
                f"{tradition}/{name}: no held-out recording remains "
                f"(only {len(recordings)} recordings)."
            )

        for rec, is_train in (
                [(r, True) for r in train_recs] +
                [(r, False) for r in test_recs]):
            tonic = read_tonic(rec)
            feature = quantize_feature(read_pitch_file(rec), tonic)
            windows = sample_windows(feature, n=n_windows, length=length, rng=rng)
            label = np.full(windows.shape[0], class_idx, dtype=np.int64)

            if is_train:
                X_train.append(windows)
                Y_train.append(label)
                split = "train"
            else:
                X_test.append(windows)
                Y_test.append(label)
                split = "held_out"

            split_info.append({
                "tradition": tradition,
                "raga": name,
                "class_idx": class_idx,
                "raga_id": raga_id,
                "recording": str(rec),
                "split": split,
                "n_windows": windows.shape[0],
            })

        if verbose:
            print(
                f"[{tradition:11s}] {name:24s} "
                f"train={len(train_recs)} held_out={len(test_recs)}"
            )

    X_train = np.concatenate(X_train) if X_train else np.empty((0, length))
    Y_train = np.concatenate(Y_train) if Y_train else np.empty(0, dtype=np.int64)
    X_test = np.concatenate(X_test) if X_test else np.empty((0, length))
    Y_test = np.concatenate(Y_test) if Y_test else np.empty(0, dtype=np.int64)

    return {
        "X_train": X_train.astype(np.int64),
        "Y_train": Y_train,
        "X_test": X_test.astype(np.int64),
        "Y_test": Y_test,
        "class_mapping": class_mapping,
        "n_classes": len(class_mapping),
        "seq_len": length,
        "n_windows": n_windows,
        "vocab_size": VOCAB_SIZE,
        "k": K,
        "split_info": split_info,
    }


if __name__ == "__main__":
    data = build_dataset(n_windows=10)
    print("X_train", data["X_train"].shape, "Y_train", data["Y_train"].shape)
    print("X_test ", data["X_test"].shape, "Y_test ", data["Y_test"].shape)
    print("n_classes", data["n_classes"])
    print("class_mapping keys", sorted(data["class_mapping"]))

