#!/usr/bin/env python3

import os
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    "/home/izpiz/coding/automatic-raga-recognition/dataset/RagaDataset"
)

OUTPUT_DIR = Path(
    "/home/izpiz/coding/automatic-raga-recognition/dataset_analysis_combined"
)

TRADITIONS = ["Carnatic", "Hindustani"]


# ============================================================
# HELPERS
# ============================================================

def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_raga_name(name):
    """
    Normalize only whitespace.
    Do NOT aggressively normalize spellings because
    we want to preserve the dataset's original labels.
    """
    if name is None:
        return None

    name = str(name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def find_files(root, extensions):
    """
    Recursively find files with given extensions.
    """
    results = []

    if not root.exists():
        return results

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            results.append(path)

    return results


def safe_relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ============================================================
# 1. LOAD RAGA MAPPINGS
# ============================================================

def load_raga_mapping(tradition):
    info_dir = DATASET_ROOT / tradition / "_info_"

    json_path = info_dir / "ragaId_to_ragaName_mapping.json"
    txt_path = info_dir / "ragaId_to_ragaName_mapping.txt"

    print(f"\n[{tradition}] Loading raga mapping...")

    if json_path.exists():
        try:
            mapping = load_json(json_path)

            # Usually dictionary: id -> name
            if isinstance(mapping, dict):
                mapping = {
                    str(k): normalize_raga_name(v)
                    for k, v in mapping.items()
                }

                print(f"Loaded {len(mapping)} raga IDs from JSON")
                return mapping

        except Exception as e:
            print(f"JSON mapping failed: {e}")

    # Fallback to TXT
    if txt_path.exists():
        mapping = {}

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Try tab separated
                parts = line.split("\t")

                if len(parts) >= 2:
                    rid = parts[0].strip()
                    name = parts[1].strip()
                    mapping[rid] = normalize_raga_name(name)
                    continue

                # Try colon separated
                if ":" in line:
                    rid, name = line.split(":", 1)
                    mapping[rid.strip()] = normalize_raga_name(name.strip())

        print(f"Loaded {len(mapping)} raga IDs from TXT")
        return mapping

    raise FileNotFoundError(
        f"No raga mapping found for {tradition}: {info_dir}"
    )


# ============================================================
# 2. LOAD TRACK -> RAGA INFORMATION
# ============================================================

def load_track_raga_metadata(tradition, raga_mapping):

    info_dir = DATASET_ROOT / tradition / "_info_"

    json_path = info_dir / "path_mbid_ragaid.json"
    txt_path = info_dir / "path_mbid_ragaid.txt"

    print(f"\n[{tradition}] Loading track/raga metadata...")

    records = []

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    if json_path.exists():

        try:
            data = load_json(json_path)

            if isinstance(data, dict):

                for path_value, value in data.items():

                    raga_id = None
                    mbid = None

                    if isinstance(value, dict):
                        raga_id = (
                            value.get("ragaId")
                            or value.get("raga_id")
                            or value.get("ragaid")
                        )

                        mbid = (
                            value.get("mbid")
                            or value.get("MBID")
                        )

                    elif isinstance(value, (list, tuple)):
                        # Attempt common formats
                        for item in value:
                            item_str = str(item)

                            if item_str in raga_mapping:
                                raga_id = item_str

                        if len(value) >= 1:
                            mbid = str(value[0])

                    else:
                        value_str = str(value)

                        if value_str in raga_mapping:
                            raga_id = value_str

                    if raga_id is None:
                        # Sometimes raga ID can be embedded in a list/string
                        text = str(value)

                        for candidate in raga_mapping:
                            if candidate in text:
                                raga_id = candidate
                                break

                    raga_name = (
                        raga_mapping.get(str(raga_id))
                        if raga_id is not None
                        else None
                    )

                    records.append({
                        "tradition": tradition,
                        "path": str(path_value),
                        "mbid": mbid,
                        "raga_id": raga_id,
                        "raga": raga_name,
                    })

                if records:
                    print(f"Loaded {len(records)} records from JSON")
                    return records

        except Exception as e:
            print(f"JSON metadata failed: {e}")

    # --------------------------------------------------------
    # TXT fallback
    # --------------------------------------------------------

    if txt_path.exists():

        with open(txt_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split("\t")

                if len(parts) < 2:
                    continue

                path_value = parts[0]

                # Search remaining fields for raga ID
                raga_id = None
                mbid = None

                for part in parts[1:]:

                    part = part.strip()

                    if part in raga_mapping:
                        raga_id = part

                    elif len(part) >= 20 and "-" in part:
                        # likely MBID
                        mbid = part

                raga_name = (
                    raga_mapping.get(raga_id)
                    if raga_id is not None
                    else None
                )

                records.append({
                    "tradition": tradition,
                    "path": path_value,
                    "mbid": mbid,
                    "raga_id": raga_id,
                    "raga": raga_name,
                })

        print(f"Loaded {len(records)} records from TXT")

    return records


# ============================================================
# 3. SCAN FEATURE FILES
# ============================================================

def scan_features(tradition):

    feature_root = DATASET_ROOT / tradition / "features"

    print(f"\n[{tradition}] Scanning features:")
    print(feature_root)

    all_files = []

    if not feature_root.exists():
        print("WARNING: feature directory does not exist")
        return pd.DataFrame()

    for path in feature_root.rglob("*"):

        if not path.is_file():
            continue

        relative = safe_relative(path, feature_root)

        suffix = path.suffix.lower()

        all_files.append({
            "tradition": tradition,
            "relative_path": relative,
            "filename": path.name,
            "extension": suffix,
            "size_bytes": path.stat().st_size,
        })

    df = pd.DataFrame(all_files)

    print(f"Total feature files: {len(df)}")

    if not df.empty:
        print("\nFile types:")
        print(df["extension"].value_counts())

    return df


# ============================================================
# 4. BUILD COMBINED DATASET
# ============================================================

def build_combined_metadata():

    all_records = []

    for tradition in TRADITIONS:

        mapping = load_raga_mapping(tradition)

        print(
            f"{tradition}: {len(mapping)} unique raga IDs"
        )

        records = load_track_raga_metadata(
            tradition,
            mapping
        )

        all_records.extend(records)

    df = pd.DataFrame(all_records)

    if df.empty:
        raise RuntimeError(
            "Could not extract any track/raga metadata."
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df["tradition"] = df["tradition"].astype(str)

    df["raga"] = df["raga"].apply(normalize_raga_name)

    # --------------------------------------------------------
    # Path information
    # --------------------------------------------------------

    df["path"] = df["path"].astype(str)

    df["filename"] = df["path"].apply(
        lambda x: Path(x).name
    )

    df["extension"] = df["path"].apply(
        lambda x: Path(x).suffix.lower()
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = OUTPUT_DIR / "combined_raga_metadata.csv"

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved combined metadata:\n{output_path}"
    )

    return df


# ============================================================
# 5. DATASET SUMMARY
# ============================================================

def dataset_summary(df):

    print("\n")
    print("=" * 100)
    print("COMBINED DATASET SUMMARY")
    print("=" * 100)

    print(f"\nTotal records: {len(df)}")

    print(
        f"Unique raga IDs: "
        f"{df['raga_id'].nunique(dropna=True)}"
    )

    print(
        f"Unique raga names: "
        f"{df['raga'].nunique(dropna=True)}"
    )

    print(
        f"Unique MBIDs: "
        f"{df['mbid'].nunique(dropna=True)}"
    )

    print("\nRecords by tradition:")
    print(df["tradition"].value_counts())

    print("\nRagas by tradition:")
    print(
        df.groupby("tradition")["raga"]
        .nunique()
    )

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nTop ragas:")
    print(
        df["raga"]
        .value_counts()
        .head(30)
    )


# ============================================================
# 6. RAGA DISTRIBUTION
# ============================================================

def analyze_raga_distribution(df):

    print("\nGenerating raga distribution...")

    counts = (
        df.groupby(["tradition", "raga"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    counts.to_csv(
        OUTPUT_DIR / "raga_distribution.csv",
        index=False
    )

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    combined = (
        df["raga"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(16, 8))

    combined.plot(kind="bar")

    plt.title(
        "Raga Distribution — Carnatic + Hindustani"
    )

    plt.xlabel("Raga")
    plt.ylabel("Number of Records")

    plt.xticks(
        rotation=75,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "raga_distribution_combined.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # By tradition
    # --------------------------------------------------------

    pivot = pd.crosstab(
        df["raga"],
        df["tradition"]
    )

    pivot.to_csv(
        OUTPUT_DIR / "raga_tradition_crosstab.csv"
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(18, 9)
    )

    ax.set_title(
        "Raga Distribution by Musical Tradition"
    )

    ax.set_xlabel("Raga")
    ax.set_ylabel("Number of Records")

    plt.xticks(
        rotation=75,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "raga_distribution_by_tradition.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 7. TRADITION DISTRIBUTION
# ============================================================

def analyze_traditions(df):

    counts = df["tradition"].value_counts()

    counts.to_csv(
        OUTPUT_DIR / "tradition_distribution.csv"
    )

    plt.figure(figsize=(8, 6))

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Carnatic vs Hindustani Dataset Distribution"
    )

    plt.xlabel("Tradition")
    plt.ylabel("Number of Records")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "tradition_distribution.png",
        dpi=200
    )

    plt.close()


# ============================================================
# 8. FEATURE INVENTORY
# ============================================================

def analyze_feature_inventory():

    frames = []

    for tradition in TRADITIONS:

        df = scan_features(tradition)

        if not df.empty:
            frames.append(df)

    if not frames:
        print("No feature files found.")
        return pd.DataFrame()

    features = pd.concat(
        frames,
        ignore_index=True
    )

    features.to_csv(
        OUTPUT_DIR / "complete_feature_inventory.csv",
        index=False
    )

    # Extension distribution

    ext_counts = (
        features
        .groupby(["tradition", "extension"])
        .size()
        .reset_index(name="count")
    )

    ext_counts.to_csv(
        OUTPUT_DIR / "feature_extension_distribution.csv",
        index=False
    )

    print("\nFeature extensions:")
    print(
        pd.crosstab(
            features["extension"],
            features["tradition"]
        )
    )

    # Plot

    pivot = pd.crosstab(
        features["extension"],
        features["tradition"]
    )

    pivot.plot(
        kind="bar",
        figsize=(12, 7)
    )

    plt.title(
        "Feature File Types — Carnatic vs Hindustani"
    )

    plt.xlabel("File Extension")
    plt.ylabel("Number of Files")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "feature_file_types.png",
        dpi=200
    )

    plt.close()

    return features


# ============================================================
# 9. FEATURE TYPE SUMMARY
# ============================================================

def feature_type_summary(features):

    if features.empty:
        return

    print("\n")
    print("=" * 100)
    print("FEATURE TYPE SUMMARY")
    print("=" * 100)

    pivot = pd.crosstab(
        features["extension"],
        features["tradition"]
    )

    print(pivot)

    # Common important feature types

    interesting = [
        ".pitch",
        ".pitchsilintrpp",
        ".tpe",
        ".tpe5msilintrpp",
        ".tonic",
        ".tonicfine",
    ]

    print("\nImportant feature types:")

    for ext in interesting:

        subset = features[
            features["extension"] == ext
        ]

        if not subset.empty:

            print(
                f"{ext:25s}: "
                f"{len(subset)} files"
            )


# ============================================================
# 10. RAGA BALANCE
# ============================================================

def analyze_balance(df):

    balance = (
        df.groupby(["tradition", "raga"])
        .size()
        .reset_index(name="samples")
    )

    balance.to_csv(
        OUTPUT_DIR / "raga_balance.csv",
        index=False
    )

    print("\nRaga balance statistics:")

    stats = (
        balance
        .groupby("tradition")["samples"]
        .agg([
            "count",
            "mean",
            "std",
            "min",
            "max"
        ])
    )

    print(stats)

    stats.to_csv(
        OUTPUT_DIR / "raga_balance_statistics.csv"
    )

    # --------------------------------------------------------
    # Histogram
    # --------------------------------------------------------

    for tradition in TRADITIONS:

        values = balance[
            balance["tradition"] == tradition
        ]["samples"]

        if len(values) == 0:
            continue

        plt.figure(figsize=(9, 6))

        plt.hist(values)

        plt.title(
            f"Samples per Raga — {tradition}"
        )

        plt.xlabel(
            "Number of Records per Raga"
        )

        plt.ylabel(
            "Number of Ragas"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR
            / f"raga_balance_{tradition.lower()}.png",
            dpi=200
        )

        plt.close()


# ============================================================
# 11. PATH DEPTH / RECORDING STRUCTURE
# ============================================================

def analyze_path_structure(df):

    df = df.copy()

    df["path_depth"] = df["path"].apply(
        lambda x: len(Path(x).parts)
    )

    path_stats = (
        df.groupby("tradition")["path_depth"]
        .describe()
    )

    path_stats.to_csv(
        OUTPUT_DIR / "path_structure_statistics.csv"
    )

    print("\nPath structure:")
    print(path_stats)

    # Components of paths

    rows = []

    for _, row in df.iterrows():

        parts = Path(row["path"]).parts

        rows.append({
            "tradition": row["tradition"],
            "path": row["path"],
            "num_components": len(parts),
        })

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "path_components.csv",
        index=False
    )


# ============================================================
# 12. SUMMARY REPORT
# ============================================================

def write_summary(df, features):

    report_path = OUTPUT_DIR / "DATASET_ANALYSIS_SUMMARY.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "RAGASENSE — COMBINED DATASET ANALYSIS\n"
        )

        f.write("=" * 80 + "\n\n")

        f.write(
            f"Dataset root:\n{DATASET_ROOT}\n\n"
        )

        f.write(
            f"Total records: {len(df)}\n"
        )

        f.write(
            f"Unique raga IDs: "
            f"{df['raga_id'].nunique(dropna=True)}\n"
        )

        f.write(
            f"Unique raga names: "
            f"{df['raga'].nunique(dropna=True)}\n"
        )

        f.write(
            f"Unique MBIDs: "
            f"{df['mbid'].nunique(dropna=True)}\n\n"
        )

        f.write(
            "RECORDS BY TRADITION\n"
        )

        f.write(
            str(df["tradition"].value_counts())
        )

        f.write("\n\n")

        f.write(
            "RAGAS BY TRADITION\n"
        )

        f.write(
            str(
                df.groupby("tradition")["raga"]
                .nunique()
            )
        )

        f.write("\n\n")

        f.write(
            "TOP RAGAS\n"
        )

        f.write(
            str(df["raga"].value_counts().head(50))
        )

        f.write("\n\n")

        f.write(
            "MISSING VALUES\n"
        )

        f.write(
            str(df.isna().sum())
        )

        f.write("\n\n")

        if not features.empty:

            f.write(
                "FEATURE FILE EXTENSIONS\n"
            )

            f.write(
                str(
                    pd.crosstab(
                        features["extension"],
                        features["tradition"]
                    )
                )
            )

            f.write("\n")

    print(
        f"\nSummary saved:\n{report_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_output_dir()

    print("=" * 100)
    print("RAGASENSE — DEEP COMBINED DATASET INSPECTION")
    print("=" * 100)

    print(f"\nDataset:")
    print(DATASET_ROOT)

    print(f"\nOutput:")
    print(OUTPUT_DIR)

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    df = build_combined_metadata()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    dataset_summary(df)

    # --------------------------------------------------------
    # Raga distribution
    # --------------------------------------------------------

    analyze_raga_distribution(df)

    # --------------------------------------------------------
    # Tradition distribution
    # --------------------------------------------------------

    analyze_traditions(df)

    # --------------------------------------------------------
    # Feature inventory
    # --------------------------------------------------------

    features = analyze_feature_inventory()

    # --------------------------------------------------------
    # Feature summary
    # --------------------------------------------------------

    feature_type_summary(features)

    # --------------------------------------------------------
    # Balance
    # --------------------------------------------------------

    analyze_balance(df)

    # --------------------------------------------------------
    # Path structure
    # --------------------------------------------------------

    analyze_path_structure(df)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    write_summary(
        df,
        features
    )

    print("\n")
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

    print(
        f"\nAll results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()