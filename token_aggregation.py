import math
import os
import json
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    RANDOM_SEED,
    DATA_DIR,
    VECTORIZER_PATH,
    MODEL_PATH,
    DATASET_PATHS,
    build_dataset_dirs,
    ensure_dirs,
    STABILITY_THRESHOLD,
    set_global_seed,
    compute_direction,
)

warnings.filterwarnings("ignore")

# =========================
# Candidate window selection parameters
# =========================
MIN_ABS_DELTA = 0.01
KEEP_PERCENT = 0.20
MIN_WINDOWS_PER_SAMPLE = 5


def add_seed_suffix(path, seed):
    base, ext = os.path.splitext(path)
    return f"{base}_seed{seed}{ext}"


def resolve_dataset_path(dataset_name, dataset_path):
    if dataset_path is not None:
        return dataset_path
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_PATHS[dataset_name]


def get_ad_score(text, vectorizer, clf):
    if not str(text).strip():
        return 0.5
    features = vectorizer.transform([str(text)])
    return float(clf.predict_proba(features)[0, 1])


def select_candidate_windows(
    windows_df,
    min_abs_delta=MIN_ABS_DELTA,
    keep_percent=KEEP_PERCENT,
    min_windows_per_sample=MIN_WINDOWS_PER_SAMPLE,
):
    required_cols = [
        "sample_id",
        "window_start",
        "window_end",
        "abs_delta_window",
        "rank",
    ]
    missing = [c for c in required_cols if c not in windows_df.columns]
    if missing:
        raise ValueError(f"window_importance missing required columns: {missing}")

    ordered_df = windows_df.sort_values(
        by=["sample_id", "abs_delta_window", "rank"],
        ascending=[True, False, True],
    )

    selected_parts = []

    for sample_id, group in ordered_df.groupby("sample_id", sort=False):
        group = group[group["abs_delta_window"] >= min_abs_delta].copy()
        if group.empty:
            continue

        n_available = len(group)
        n_top = math.ceil(keep_percent * n_available)
        n_keep = min(n_available, max(n_top, min_windows_per_sample))

        group = group.sort_values(
            by=["abs_delta_window", "rank"],
            ascending=[False, True],
        ).head(n_keep)

        selected_parts.append(group)

    if selected_parts:
        return pd.concat(selected_parts).reset_index(drop=True)
    return pd.DataFrame(columns=windows_df.columns)


def make_occurrence_id(sample_id, token_position):
    return f"{sample_id}:{token_position}"


def build_candidate_occurrence_map(candidate_windows):
    """
    Build a map:
        occurrence_id -> metadata
    only for token occurrences covered by selected candidate windows.
    """
    occurrence_map = {}

    for _, row in candidate_windows.iterrows():
        sample_id = row["sample_id"]
        start = int(row["window_start"])
        end = int(row["window_end"])

        for token_position in range(start, end):
            occurrence_id = make_occurrence_id(sample_id, token_position)

            if occurrence_id not in occurrence_map:
                occurrence_map[occurrence_id] = {
                    "sample_id": sample_id,
                    "token_position": token_position,
                    "covering_windows": [],
                }

            occurrence_map[occurrence_id]["covering_windows"].append(
                {
                    "window_rank": int(row["rank"]) if "rank" in row else None,
                    "window_start": start,
                    "window_end": end,
                    "window_text": row.get("window_text", ""),
                    "window_direction": row.get("direction_window", None),
                    "delta_window": float(row.get("delta_window", 0.0)),
                    "abs_delta_window": float(row.get("abs_delta_window", 0.0)),
                }
            )

    return occurrence_map


def analyze_tokens(df, candidate_windows, vectorizer, clf, seed, dataset_name):
    """
    Analyze only token occurrences that are covered by candidate windows.
    """
    occurrence_map = build_candidate_occurrence_map(candidate_windows)
    all_records = []

    for idx, row in df.iterrows():
        sample_id = row.get("sample_id", idx)
        text = str(row.get("Speech", "")).strip()
        if not text:
            continue

        tokens = text.split()
        if not tokens:
            continue

        original_prob = get_ad_score(text, vectorizer, clf)

        for token_position, token in enumerate(tokens):
            occurrence_id = make_occurrence_id(sample_id, token_position)
            if occurrence_id not in occurrence_map:
                continue

            masked_tokens = tokens[:token_position] + tokens[token_position + 1 :]
            masked_text = " ".join(masked_tokens)

            score_without_token = get_ad_score(masked_text, vectorizer, clf)

            delta_token = original_prob - score_without_token
            abs_delta_token = abs(delta_token)
            direction_token = compute_direction(delta_token)

            covering_windows = occurrence_map[occurrence_id]["covering_windows"]

            representative = max(
                covering_windows,
                key=lambda w: (w["abs_delta_window"], -(w["window_rank"] or 10**9)),
            )

            all_records.append(
                {
                    "dataset": dataset_name,
                    "seed": seed,
                    "sample_id": sample_id,
                    "token": token,
                    "token_position": token_position,
                    "occurrence_id": occurrence_id,
                    "window_rank": representative["window_rank"],
                    "window_start": representative["window_start"],
                    "window_end": representative["window_end"],
                    "window_text": representative["window_text"],
                    "window_direction": representative["window_direction"],
                    "representative_delta_window": representative["delta_window"],
                    "representative_abs_delta_window": representative[
                        "abs_delta_window"
                    ],
                    "cover_count": len(covering_windows),
                    "score_without_token": score_without_token,
                    "delta_token": delta_token,
                    "abs_delta_token": abs_delta_token,
                    "direction_token": direction_token,
                }
            )

    return pd.DataFrame(all_records)


def aggregate_tokens(token_df):
    if token_df.empty:
        return pd.DataFrame(
            columns=[
                "token",
                "n_occurrences",
                "raw_importance",
                "signed_mean_delta",
                "mean_importance",
                "log_adjusted_importance",
                "ad_support_count",
                "control_support_count",
                "neutral_count",
                "ad_support_ratio",
                "primary_direction",
                "direction_stable",
            ]
        )

    grouped = token_df.groupby("token", dropna=False)

    aggregated = grouped.agg(
        n_occurrences=("occurrence_id", "nunique"),
        raw_importance=("abs_delta_token", "sum"),
        signed_mean_delta=("delta_token", "mean"),
        ad_support_count=("direction_token", lambda s: int((s == "AD_support").sum())),
        control_support_count=(
            "direction_token",
            lambda s: int((s == "Control_support").sum()),
        ),
        neutral_count=("direction_token", lambda s: int((s == "Neutral").sum())),
    ).reset_index()

    aggregated["mean_importance"] = (
        aggregated["raw_importance"] / aggregated["n_occurrences"]
    )

    aggregated["log_adjusted_importance"] = aggregated["raw_importance"] / np.log1p(
        aggregated["n_occurrences"]
    )

    directional_total = (
        aggregated["ad_support_count"] + aggregated["control_support_count"]
    )

    aggregated["ad_support_ratio"] = np.where(
        directional_total > 0,
        aggregated["ad_support_count"] / directional_total,
        0.5,
    )

    aggregated["primary_direction"] = np.where(
        aggregated["ad_support_ratio"] >= STABILITY_THRESHOLD,
        "AD_support",
        np.where(
            aggregated["ad_support_ratio"] <= (1 - STABILITY_THRESHOLD),
            "Control_support",
            "mixed",
        ),
    )

    aggregated["direction_stable"] = aggregated["primary_direction"].isin(
        ["AD_support", "Control_support"]
    )

    aggregated = aggregated.sort_values(
        by=["log_adjusted_importance", "raw_importance", "token"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return aggregated


def main(
    seed=RANDOM_SEED,
    output_with_seed=False,
    dataset_name="test",
    dataset_path=None,
):
    set_global_seed(seed)

    dataset_path = resolve_dataset_path(dataset_name, dataset_path)
    dirs = build_dataset_dirs("token_aggregation", dataset_name)
    ensure_dirs(dirs, keys=["csv"])

    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)

    df = pd.read_csv(dataset_path)
    df["Speech"] = df["Speech"].fillna("")

    window_path = Path(DATA_DIR) / "window_extraction" / dataset_name / "csv" / "window_importance.csv"

    if output_with_seed:
        window_path = Path(add_seed_suffix(str(window_path), seed))

    windows_df = pd.read_csv(window_path)

    candidate_windows = select_candidate_windows(windows_df)

    print("=" * 80)
    print(f"Token Analysis | dataset={dataset_name} | seed={seed}")
    print("=" * 80)
    print(f"Loaded windows: {window_path}")
    print(f"Selected candidate windows: {len(candidate_windows)}")

    token_df = analyze_tokens(
        df, candidate_windows, vectorizer, clf, seed, dataset_name
    )
    agg_df = aggregate_tokens(token_df)

    selected_windows_path = Path(dirs["csv"]) / "candidate_windows_selected.csv"
    all_path = Path(dirs["csv"]) / "token_attribution_all.csv"
    agg_path = Path(dirs["csv"]) / "token_attribution_aggregated.csv"

    if output_with_seed:
        selected_windows_path = Path(add_seed_suffix(str(selected_windows_path), seed))
        all_path = Path(add_seed_suffix(str(all_path), seed))
        agg_path = Path(add_seed_suffix(str(agg_path), seed))

    candidate_windows.to_csv(selected_windows_path, index=False)
    token_df.to_csv(all_path, index=False)
    agg_df.to_csv(agg_path, index=False)

    print(f"Saved: {selected_windows_path}")
    print(f"Saved: {all_path}")
    print(f"Saved: {agg_path}")

    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_candidate_windows": len(candidate_windows),
        "n_token_occurrences": len(token_df),
        "n_token_types": len(agg_df),
    }


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASET_PATHS.items():
        main(
            seed=RANDOM_SEED,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
