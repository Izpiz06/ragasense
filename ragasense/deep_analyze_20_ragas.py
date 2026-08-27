#!/usr/bin/env python3

"""
DeepSRGM — Deep analysis of the selected 20-raga dataset.

Dataset structure expected:

RagaDataset/
├── Carnatic/
│   ├── _info_/
│   │   ├── path_mbid_ragaid.json
│   │   ├── ragaId_to_ragaName_mapping.json
│   │   └── ...
│   └── features/
│       └── <raga-id>/
│           └── ...
│
└── Hindustani/
    ├── _info_/
    │   ├── path_mbid_ragaid.json
    │   ├── ragaId_to_ragaName_mapping.json
    │   └── ...
    └── features/
        └── <raga-id>/
            └── ...

No audio is required.

Outputs:
dataset_analysis_20_ragas/
├── selected_20_ragas_metadata.csv
├── selected_20_ragas_feature_analysis.csv
├── raga_summary.csv
├── tradition_summary.csv
├── feature_availability.csv
├── tonic_statistics.csv
├── pitch_statistics.csv
├── dataset_summary.txt
└── figures/
    ├── recordings_per_raga.png
    ├── recordings_by_tradition.png
    ├── feature_availability.png
    ├── pitch_duration_by_raga.png
    ├── pitch_range_by_raga.png
    ├── tonic_distribution.png
    ├── pitch_density.png
    └── dataset_balance.png
"""

from pathlib import Path
import json
import re
import csv
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/izpiz/coding/automatic-raga-recognition/dataset/RagaDataset"
)

OUTPUT_DIR = Path(
    "/home/izpiz/coding/automatic-raga-recognition/"
    "dataset_analysis_20_ragas"
)

FIGURE_DIR = OUTPUT_DIR / "figures"


# ------------------------------------------------------------
# Selected ragas
# ------------------------------------------------------------

SELECTED_RAGAS = {
    "Carnatic": [
        "Aṭāna",
        "Bhairavi",
        "Kalyāṇi",
        "Kāpi",
        "Kāṁbhōji",
        "Kēdāragauḷa",
        "Mōhanaṁ",
        "Rītigauḷa",
        "Tōḍi",
        "Śankarābharaṇaṁ",
    ],

    "Hindustani": [
        "Bhairav",
        "Jōg",
        "Bihāg",
        "Bilāsakhānī tōḍī",
        "Darbāri",
        "Khamāj",
        "Dēś",
        "Miyān malhār",
        "Yaman kalyāṇ",
        "Śrī",
    ],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(value):
    """
    Normalize strings sufficiently for matching names while
    preserving the original names in the output.
    """
    if value is None:
        return ""

    value = str(value)

    replacements = {
        "ā": "a",
        "Ā": "a",
        "ī": "i",
        "Ī": "i",
        "ū": "u",
        "Ū": "u",
        "ē": "e",
        "Ē": "e",
        "ō": "o",
        "Ō": "o",
        "ṁ": "m",
        "ṃ": "m",
        "ṅ": "n",
        "ñ": "n",
        "ṇ": "n",
        "ṭ": "t",
        "ḍ": "d",
        "ṛ": "r",
        "ś": "s",
        "Ś": "s",
        "ṣ": "s",
        "ḷ": "l",
        "’": "'",
        "–": "-",
        "—": "-",
    }

    for a, b in replacements.items():
        value = value.replace(a, b)

    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    return value


def ensure_output_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def find_file(directory, filename):
    """
    Find a file recursively.
    """
    matches = list(directory.rglob(filename))

    if matches:
        return matches[0]

    return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x):
    try:
        value = float(x)

        if math.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# INFO FILES
# ============================================================

def load_raga_mapping(tradition_dir):
    """
    Load ragaId_to_ragaName_mapping.json
    """

    info_dir = tradition_dir / "_info_"

    mapping_path = info_dir / "ragaId_to_ragaName_mapping.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{mapping_path}"
        )

    mapping = load_json(mapping_path)

    # Usually:
    # {
    #    "uuid": "Raga name"
    # }
    #
    # But handle nested structures as well.

    result = {}

    if isinstance(mapping, dict):

        for key, value in mapping.items():

            if isinstance(value, str):
                result[str(key)] = value

            elif isinstance(value, dict):

                # Try common possible keys
                name = (
                    value.get("name")
                    or value.get("raga")
                    or value.get("ragaName")
                )

                if name:
                    result[str(key)] = str(name)

    return result


def load_path_mapping(tradition_dir):
    """
    Load path_mbid_ragaid.json

    This maps recordings/files to raga IDs.
    """

    info_dir = tradition_dir / "_info_"

    mapping_path = info_dir / "path_mbid_ragaid.json"

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Could not find:\n{mapping_path}"
        )

    mapping = load_json(mapping_path)

    return mapping


# ============================================================
# PATH MAPPING PARSER
# ============================================================

def extract_records_from_mapping(mapping):
    """
    Convert potentially different JSON structures into:

        [
            {
                "path": ...,
                "mbid": ...,
                "raga_id": ...
            }
        ]

    The RagaDataset metadata format can vary between versions,
    so this function intentionally handles several structures.
    """

    records = []

    if isinstance(mapping, dict):

        # Case 1:
        # path -> [mbid, ragaid]
        for key, value in mapping.items():

            if isinstance(value, (list, tuple)):

                if len(value) >= 2:

                    records.append({
                        "path": str(key),
                        "mbid": str(value[0]),
                        "raga_id": str(value[1]),
                    })

            elif isinstance(value, dict):

                path = (
                    value.get("path")
                    or value.get("filepath")
                    or value.get("file")
                    or key
                )

                mbid = (
                    value.get("mbid")
                    or value.get("MBID")
                    or value.get("musicbrainz_id")
                    or ""
                )

                raga_id = (
                    value.get("ragaid")
                    or value.get("raga_id")
                    or value.get("ragaId")
                    or value.get("raga")
                )

                if raga_id is not None:

                    records.append({
                        "path": str(path),
                        "mbid": str(mbid),
                        "raga_id": str(raga_id),
                    })

    elif isinstance(mapping, list):

        for item in mapping:

            if isinstance(item, dict):

                path = (
                    item.get("path")
                    or item.get("filepath")
                    or item.get("file")
                    or ""
                )

                mbid = (
                    item.get("mbid")
                    or item.get("MBID")
                    or ""
                )

                raga_id = (
                    item.get("ragaid")
                    or item.get("raga_id")
                    or item.get("ragaId")
                    or item.get("raga")
                )

                if raga_id is not None:

                    records.append({
                        "path": str(path),
                        "mbid": str(mbid),
                        "raga_id": str(raga_id),
                    })

    return records


# ============================================================
# FEATURE DISCOVERY
# ============================================================

def discover_feature_files(feature_dir):
    """
    Find feature files.

    We care particularly about:
        .pitch
        .pitchSilIntrpPP
        .tonic
        .tonicFine

    We also retain all other feature types for inventory.
    """

    files = []

    if not feature_dir.exists():
        return files

    for path in feature_dir.rglob("*"):

        if path.is_file():

            files.append(path)

    return files


def identify_feature_type(path):
    name = path.name

    if name.endswith(".pitchSilIntrpPP"):
        return "pitch_post_processed"

    if name.endswith(".pitch"):
        return "pitch"

    if name.endswith(".tonicFine"):
        return "tonic_fine"

    if name.endswith(".tonic"):
        return "tonic"

    if name.endswith(".flatSegNyas"):
        return "nyas_segments"

    if name.endswith(".taniSegKNN"):
        return "tani_segments"

    return path.suffix.lower().lstrip(".") or "unknown"


# ============================================================
# PITCH READING
# ============================================================

def read_pitch_file(path):
    """
    Attempt to read pitch files robustly.

    DeepSRGM-style pitch files commonly contain time/pitch
    information, but formatting can vary.

    Returns:
        times
        pitches
    """

    try:

        # Try whitespace separated
        arr = np.loadtxt(path)

    except Exception:

        try:
            arr = np.genfromtxt(
                path,
                delimiter=",",
                comments="#"
            )

        except Exception:
            return np.array([]), np.array([])

    if arr.size == 0:
        return np.array([]), np.array([])

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    # Common cases:
    #
    # time pitch
    # pitch
    #
    if arr.shape[1] >= 2:

        times = arr[:, 0]
        pitches = arr[:, 1]

    else:

        pitches = arr[:, 0]

        # Approximate 5ms frame spacing
        times = np.arange(len(pitches)) * 0.005

    times = np.asarray(times, dtype=float)
    pitches = np.asarray(pitches, dtype=float)

    valid = (
        np.isfinite(times)
        & np.isfinite(pitches)
    )

    times = times[valid]
    pitches = pitches[valid]

    return times, pitches


# ============================================================
# TONIC READING
# ============================================================

def read_tonic_file(path):

    try:

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Extract first numerical value
        match = re.search(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            text
        )

        if match:
            return float(match.group())

    except Exception:
        pass

    return np.nan


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def analyze_pitch(path):

    result = {
        "pitch_file": str(path),
        "pitch_rows": 0,
        "valid_pitch_frames": 0,
        "invalid_pitch_frames": 0,
        "duration_seconds": np.nan,
        "pitch_min": np.nan,
        "pitch_max": np.nan,
        "pitch_mean": np.nan,
        "pitch_median": np.nan,
        "pitch_std": np.nan,
        "voiced_ratio": np.nan,
    }

    times, pitches = read_pitch_file(path)

    result["pitch_rows"] = len(pitches)

    if len(pitches) == 0:
        return result

    # Consider <= 0 as unvoiced in most pitch tracks
    valid_pitch = pitches[
        np.isfinite(pitches) & (pitches > 0)
    ]

    result["valid_pitch_frames"] = len(valid_pitch)
    result["invalid_pitch_frames"] = (
        len(pitches) - len(valid_pitch)
    )

    if len(times) > 0:

        result["duration_seconds"] = (
            np.nanmax(times) - np.nanmin(times)
        )

    if len(valid_pitch) > 0:

        result["pitch_min"] = np.min(valid_pitch)
        result["pitch_max"] = np.max(valid_pitch)
        result["pitch_mean"] = np.mean(valid_pitch)
        result["pitch_median"] = np.median(valid_pitch)
        result["pitch_std"] = np.std(valid_pitch)

        result["voiced_ratio"] = (
            len(valid_pitch) / len(pitches)
        )

    return result


# ============================================================
# BUILD DATASET TABLE
# ============================================================

def build_tradition_table(tradition):

    tradition_dir = DATASET_ROOT / tradition

    print()
    print("=" * 90)
    print(f"BUILDING DATASET TABLE — {tradition}")
    print("=" * 90)

    mapping = load_path_mapping(tradition_dir)
    raga_mapping = load_raga_mapping(tradition_dir)

    records = extract_records_from_mapping(mapping)

    print(f"Mapping records found: {len(records)}")
    print(f"Raga IDs in mapping: {len(raga_mapping)}")

    rows = []

    feature_root = tradition_dir / "features"

    # Build normalized raga lookup
    normalized_ragas = {
        normalize_text(name): (rid, name)
        for rid, name in raga_mapping.items()
    }

    for record in records:

        raga_id = str(record["raga_id"])

        raga_name = raga_mapping.get(
            raga_id,
            raga_id
        )

        # Match selected raga
        selected = None

        normalized_name = normalize_text(raga_name)

        for target in SELECTED_RAGAS[tradition]:

            if normalize_text(target) == normalized_name:
                selected = target
                break

        if selected is None:
            continue

        # Raga-specific feature directory
        raga_feature_dir = feature_root / raga_id

        # Sometimes raga IDs in metadata may not exactly match
        # directory naming. Fall back to recursive search.
        if not raga_feature_dir.exists():

            candidates = [
                p for p in feature_root.iterdir()
                if p.is_dir() and p.name == raga_id
            ]

            if candidates:
                raga_feature_dir = candidates[0]

        rows.append({
            "tradition": tradition,
            "raga": selected,
            "raga_id": raga_id,
            "mbid": record["mbid"],
            "metadata_path": record["path"],
            "feature_dir": str(raga_feature_dir),
        })

    df = pd.DataFrame(rows)

    print(f"{tradition}: {len(df)} recordings")

    if len(df) > 0:

        print()
        print("Selected raga distribution:")

        print(
            df["raga"]
            .value_counts()
            .sort_index()
            .to_string()
        )

    return df


# ============================================================
# FIND RECORDING FEATURES
# ============================================================

def attach_features(df):

    all_rows = []

    for _, row in df.iterrows():

        feature_dir = Path(row["feature_dir"])

        feature_files = discover_feature_files(feature_dir)

        # Group files by basename.
        #
        # Example:
        # foo.pitch
        # foo.pitchSilIntrpPP
        # foo.tonic
        # foo.tonicFine

        groups = {}

        for path in feature_files:

            feature_type = identify_feature_type(path)

            # Remove known suffixes
            stem = path.name

            for suffix in [
                ".pitchSilIntrpPP",
                ".flatSegNyas",
                ".taniSegKNN",
                ".tonicFine",
                ".pitch",
                ".tonic",
            ]:

                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
                    break

            groups.setdefault(stem, {})
            groups[stem][feature_type] = path

        # If no grouped files were found, still preserve
        # metadata record.
        if not groups:

            all_rows.append({
                **row.to_dict(),
                "feature_record": "",
                "pitch_path": "",
                "pitch_post_processed_path": "",
                "tonic_path": "",
                "tonic_fine_path": "",
            })

            continue

        for feature_record, features in groups.items():

            all_rows.append({
                **row.to_dict(),
                "feature_record": feature_record,
                "pitch_path": str(
                    features.get("pitch", "")
                ),
                "pitch_post_processed_path": str(
                    features.get("pitch_post_processed", "")
                ),
                "tonic_path": str(
                    features.get("tonic", "")
                ),
                "tonic_fine_path": str(
                    features.get("tonic_fine", "")
                ),
            })

    return pd.DataFrame(all_rows)


# ============================================================
# ANALYZE FEATURES
# ============================================================

def analyze_dataset(df):

    rows = []

    for index, row in df.iterrows():

        if index % 100 == 0:
            print(
                f"Analyzing feature record "
                f"{index}/{len(df)}"
            )

        result = row.to_dict()

        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        pitch_path = row.get(
            "pitch_post_processed_path",
            ""
        )

        if not pitch_path or pitch_path == "":

            pitch_path = row.get(
                "pitch_path",
                ""
            )

        if pitch_path and Path(pitch_path).exists():

            pitch_stats = analyze_pitch(
                Path(pitch_path)
            )

            result.update(pitch_stats)

        else:

            result.update({
                "pitch_file": "",
                "pitch_rows": 0,
                "valid_pitch_frames": 0,
                "invalid_pitch_frames": 0,
                "duration_seconds": np.nan,
                "pitch_min": np.nan,
                "pitch_max": np.nan,
                "pitch_mean": np.nan,
                "pitch_median": np.nan,
                "pitch_std": np.nan,
                "voiced_ratio": np.nan,
            })

        # ----------------------------------------------------
        # Tonic
        # ----------------------------------------------------

        tonic_path = row.get(
            "tonic_fine_path",
            ""
        )

        if not tonic_path:

            tonic_path = row.get(
                "tonic_path",
                ""
            )

        if tonic_path and Path(tonic_path).exists():

            result["tonic"] = read_tonic_file(
                Path(tonic_path)
            )

        else:

            result["tonic"] = np.nan

        # ----------------------------------------------------
        # Feature availability
        # ----------------------------------------------------

        result["has_pitch"] = bool(
            row.get("pitch_path")
            and Path(row["pitch_path"]).exists()
        )

        result["has_pitch_post_processed"] = bool(
            row.get("pitch_post_processed_path")
            and Path(
                row["pitch_post_processed_path"]
            ).exists()
        )

        result["has_tonic"] = bool(
            row.get("tonic_path")
            and Path(row["tonic_path"]).exists()
        )

        result["has_tonic_fine"] = bool(
            row.get("tonic_fine_path")
            and Path(
                row["tonic_fine_path"]
            ).exists()
        )

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# SUMMARIES
# ============================================================

def generate_summaries(df):

    if df.empty:
        print("\nWARNING: No records found.")
        return

    # --------------------------------------------------------
    # Raga summary
    # --------------------------------------------------------

    raga_summary = (
        df.groupby(
            ["tradition", "raga"],
            dropna=False
        )
        .agg(
            recordings=("mbid", "nunique"),
            feature_records=("feature_record", "count"),
            pitch_files=("has_pitch", "sum"),
            post_processed_pitch_files=(
                "has_pitch_post_processed",
                "sum"
            ),
            tonic_files=("has_tonic", "sum"),
            tonic_fine_files=("has_tonic_fine", "sum"),
            total_pitch_frames=(
                "valid_pitch_frames",
                "sum"
            ),
            mean_duration_seconds=(
                "duration_seconds",
                "mean"
            ),
            mean_voiced_ratio=(
                "voiced_ratio",
                "mean"
            ),
            mean_pitch=(
                "pitch_mean",
                "mean"
            ),
            mean_pitch_std=(
                "pitch_std",
                "mean"
            ),
        )
        .reset_index()
    )

    raga_summary.to_csv(
        OUTPUT_DIR / "raga_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Tradition summary
    # --------------------------------------------------------

    tradition_summary = (
        df.groupby("tradition")
        .agg(
            feature_records=("feature_record", "count"),
            unique_ragas=("raga", "nunique"),
            recordings=("mbid", "nunique"),
            pitch_files=("has_pitch", "sum"),
            post_processed_pitch_files=(
                "has_pitch_post_processed",
                "sum"
            ),
            tonic_files=("has_tonic", "sum"),
            tonic_fine_files=("has_tonic_fine", "sum"),
            total_pitch_frames=(
                "valid_pitch_frames",
                "sum"
            ),
            mean_duration_seconds=(
                "duration_seconds",
                "mean"
            ),
            mean_voiced_ratio=(
                "voiced_ratio",
                "mean"
            ),
        )
        .reset_index()
    )

    tradition_summary.to_csv(
        OUTPUT_DIR / "tradition_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # Feature availability
    # --------------------------------------------------------

    feature_rows = []

    for tradition in SELECTED_RAGAS:

        subset = df[
            df["tradition"] == tradition
        ]

        total = len(subset)

        for feature in [
            "has_pitch",
            "has_pitch_post_processed",
            "has_tonic",
            "has_tonic_fine",
        ]:

            count = int(subset[feature].sum())

            feature_rows.append({
                "tradition": tradition,
                "feature": feature,
                "available": count,
                "total": total,
                "percentage": (
                    100 * count / total
                    if total > 0 else 0
                ),
            })

    feature_df = pd.DataFrame(feature_rows)

    feature_df.to_csv(
        OUTPUT_DIR / "feature_availability.csv",
        index=False
    )

    # --------------------------------------------------------
    # Tonic statistics
    # --------------------------------------------------------

    tonic_df = (
        df.groupby(
            ["tradition", "raga"]
        )["tonic"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    tonic_df.to_csv(
        OUTPUT_DIR / "tonic_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Pitch statistics
    # --------------------------------------------------------

    pitch_df = (
        df.groupby(
            ["tradition", "raga"]
        )
        .agg(
            records=("feature_record", "count"),
            valid_frames=("valid_pitch_frames", "sum"),
            mean_duration=("duration_seconds", "mean"),
            median_duration=("duration_seconds", "median"),
            mean_pitch=("pitch_mean", "mean"),
            median_pitch=("pitch_median", "mean"),
            pitch_std=("pitch_std", "mean"),
            mean_voiced_ratio=("voiced_ratio", "mean"),
            min_pitch=("pitch_min", "min"),
            max_pitch=("pitch_max", "max"),
        )
        .reset_index()
    )

    pitch_df.to_csv(
        OUTPUT_DIR / "pitch_statistics.csv",
        index=False
    )


# ============================================================
# FIGURES
# ============================================================

def save_figures(df):

    if df.empty:
        print("No data available for figures.")
        return

    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "axes.grid": True,
    })

    # --------------------------------------------------------
    # 1. Recordings per raga
    # --------------------------------------------------------

    counts = (
        df.groupby(
            ["tradition", "raga"]
        )["mbid"]
        .nunique()
        .reset_index()
    )

    plt.figure()

    labels = (
        counts["tradition"]
        + " — "
        + counts["raga"]
    )

    plt.bar(
        labels,
        counts["mbid"]
    )

    plt.xticks(
        rotation=70,
        ha="right"
    )

    plt.ylabel("Number of recordings")
    plt.title("Recordings per Selected Raga")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "recordings_per_raga.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 2. Tradition balance
    # --------------------------------------------------------

    tradition_counts = (
        df.groupby("tradition")["mbid"]
        .nunique()
    )

    plt.figure()

    plt.bar(
        tradition_counts.index,
        tradition_counts.values
    )

    plt.ylabel("Number of recordings")
    plt.title("Dataset Balance: Carnatic vs Hindustani")
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "recordings_by_tradition.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 3. Feature availability
    # --------------------------------------------------------

    features = [
        "has_pitch",
        "has_pitch_post_processed",
        "has_tonic",
        "has_tonic_fine",
    ]

    percentages = []

    for feature in features:

        percentages.append(
            100 * df[feature].mean()
        )

    plt.figure()

    plt.bar(
        features,
        percentages
    )

    plt.ylabel("Availability (%)")
    plt.title("Feature Availability")
    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.ylim(0, 105)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "feature_availability.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # 4. Duration by raga
    # --------------------------------------------------------

    duration_df = df.dropna(
        subset=["duration_seconds"]
    )

    if not duration_df.empty:

        groups = []

        labels = []

        for tradition, raga in (
            duration_df[
                ["tradition", "raga"]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        ):

            values = duration_df[
                (duration_df["tradition"] == tradition)
                & (duration_df["raga"] == raga)
            ]["duration_seconds"].values

            if len(values):

                groups.append(values)

                labels.append(
                    f"{tradition}\n{raga}"
                )

        plt.figure(
            figsize=(15, 8)
        )

        plt.boxplot(
            groups,
            tick_labels=labels,
            showfliers=False
        )

        plt.ylabel("Duration (seconds)")
        plt.title("Recording/Pitch Duration Distribution")
        plt.xticks(
            rotation=70
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "pitch_duration_by_raga.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # 5. Pitch range
    # --------------------------------------------------------

    pitch_range = df.dropna(
        subset=[
            "pitch_min",
            "pitch_max"
        ]
    ).copy()

    if not pitch_range.empty:

        pitch_range["range"] = (
            pitch_range["pitch_max"]
            - pitch_range["pitch_min"]
        )

        grouped = (
            pitch_range
            .groupby(
                ["tradition", "raga"]
            )["range"]
            .mean()
            .sort_values()
        )

        plt.figure(
            figsize=(14, 8)
        )

        labels = [
            f"{a}\n{b}"
            for a, b in grouped.index
        ]

        plt.bar(
            labels,
            grouped.values
        )

        plt.ylabel("Mean pitch range")
        plt.title("Average Pitch Range by Raga")

        plt.xticks(
            rotation=70
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "pitch_range_by_raga.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # 6. Tonic distribution
    # --------------------------------------------------------

    tonic_df = df.dropna(
        subset=["tonic"]
    )

    if not tonic_df.empty:

        plt.figure()

        for tradition in tonic_df["tradition"].unique():

            subset = tonic_df[
                tonic_df["tradition"] == tradition
            ]

            plt.hist(
                subset["tonic"],
                bins=20,
                alpha=0.6,
                label=tradition
            )

        plt.xlabel("Tonic")
        plt.ylabel("Number of feature records")
        plt.title("Tonic Distribution")
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "tonic_distribution.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # 7. Pitch density
    # --------------------------------------------------------

    density_df = df[
        (df["duration_seconds"] > 0)
        & (df["valid_pitch_frames"] > 0)
    ].copy()

    if not density_df.empty:

        density_df["pitch_frames_per_second"] = (
            density_df["valid_pitch_frames"]
            / density_df["duration_seconds"]
        )

        grouped = (
            density_df
            .groupby(
                ["tradition", "raga"]
            )["pitch_frames_per_second"]
            .mean()
        )

        plt.figure(
            figsize=(14, 8)
        )

        labels = [
            f"{a}\n{b}"
            for a, b in grouped.index
        ]

        plt.bar(
            labels,
            grouped.values
        )

        plt.ylabel("Pitch frames / second")
        plt.title("Pitch Feature Density")

        plt.xticks(
            rotation=70
        )

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / "pitch_density.png",
            dpi=200
        )

        plt.close()

    # --------------------------------------------------------
    # 8. Dataset balance heatmap-like figure
    # --------------------------------------------------------

    pivot = (
        df.groupby(
            ["tradition", "raga"]
        )["mbid"]
        .nunique()
        .unstack(0)
        .fillna(0)
    )

    plt.figure(
        figsize=(8, 10)
    )

    plt.imshow(
        pivot.values,
        aspect="auto"
    )

    plt.colorbar(
        label="Number of recordings"
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns
    )

    plt.title(
        "Selected Raga Dataset Balance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "dataset_balance.png",
        dpi=200
    )

    plt.close()


# ============================================================
# TEXT SUMMARY
# ============================================================

def write_text_summary(df):

    summary_path = OUTPUT_DIR / "dataset_summary.txt"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "DEEPSRGM — 20 RAGA DATASET ANALYSIS\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write(
            f"Dataset root:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Selected ragas: 20\n"
        )

        f.write(
            f"Total feature records: {len(df)}\n\n"
        )

        if df.empty:
            f.write(
                "WARNING: No feature records were found.\n"
            )
            return

        f.write(
            f"Traditions: "
            f"{df['tradition'].nunique()}\n"
        )

        f.write(
            f"Ragas found: "
            f"{df['raga'].nunique()}\n\n"
        )

        f.write(
            "RAGA DISTRIBUTION\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        counts = (
            df.groupby(
                ["tradition", "raga"]
            )["mbid"]
            .nunique()
        )

        for (tradition, raga), count in counts.items():

            f.write(
                f"{tradition:12s} | "
                f"{raga:30s} | "
                f"{count} recordings\n"
            )

        f.write("\n")

        f.write(
            "FEATURE AVAILABILITY\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for feature in [
            "has_pitch",
            "has_pitch_post_processed",
            "has_tonic",
            "has_tonic_fine",
        ]:

            count = int(df[feature].sum())

            percentage = (
                100 * count / len(df)
            )

            f.write(
                f"{feature:30s}: "
                f"{count}/{len(df)} "
                f"({percentage:.2f}%)\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 100)
    print(
        "DEEPSRGM — 20 RAGA DATASET DEEP ANALYSIS"
    )
    print("=" * 100)

    print()
    print(
        f"Dataset:\n{DATASET_ROOT}"
    )

    print()
    print(
        f"Output:\n{OUTPUT_DIR}"
    )

    ensure_output_dirs()

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nDataset does not exist:\n{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    all_metadata = []

    for tradition in [
        "Carnatic",
        "Hindustani"
    ]:

        tradition_df = build_tradition_table(
            tradition
        )

        if not tradition_df.empty:
            all_metadata.append(
                tradition_df
            )

    if not all_metadata:

        print(
            "\nERROR: No selected ragas were found."
        )

        return

    metadata_df = pd.concat(
        all_metadata,
        ignore_index=True
    )

    metadata_path = (
        OUTPUT_DIR
        / "selected_20_ragas_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False
    )

    print()
    print(
        f"Saved metadata:\n{metadata_path}"
    )

    # --------------------------------------------------------
    # Attach features
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("DISCOVERING FEATURE FILES")
    print("=" * 90)

    feature_df = attach_features(
        metadata_df
    )

    print(
        f"\nFeature records discovered: "
        f"{len(feature_df)}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("ANALYZING RECORDINGS AND FEATURES")
    print("=" * 90)

    analysis_df = analyze_dataset(
        feature_df
    )

    analysis_path = (
        OUTPUT_DIR
        / "selected_20_ragas_feature_analysis.csv"
    )

    analysis_df.to_csv(
        analysis_path,
        index=False
    )

    print(
        f"\nSaved:\n{analysis_path}"
    )

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("GENERATING SUMMARIES")
    print("=" * 90)

    generate_summaries(
        analysis_df
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("GENERATING FIGURES")
    print("=" * 90)

    save_figures(
        analysis_df
    )

    # --------------------------------------------------------
    # Text summary
    # --------------------------------------------------------

    write_text_summary(
        analysis_df
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

    print(
        f"\nOutput directory:\n{OUTPUT_DIR}"
    )

    print("\nGenerated files:")

    for path in sorted(
        OUTPUT_DIR.rglob("*")
    ):

        if path.is_file():

            print(
                f"  {path.relative_to(OUTPUT_DIR)}"
            )

    print()
    print("Important:")
    print(
        "This analysis does NOT require audio recordings."
    )

    print(
        "It works from the RagaDataset metadata mappings "
        "and feature files."
    )


if __name__ == "__main__":
    main()