#!/usr/bin/env python3
"""
seed_summary.py

Aggregate multi-seed outputs for the AD speech-text pipeline.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_BASE_DIR = "data_Test AD"
DEFAULT_DATASETS = ["test", "ad", "control"]
DEFAULT_TOPK = 20
DEFAULT_MIN_SEEDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed outputs.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base project directory. Default: data_Test AD",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets to summarize, e.g. test,ad,control",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOPK,
        help="Top-k windows used for overlap and stable-window export.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=DEFAULT_MIN_SEEDS,
        help="Minimum distinct seeds required to generate outputs.",
    )
    parser.add_argument(
        "--stable-threshold",
        type=float,
        default=0.9,
        help="Consistency threshold for stable_window / stable_token.",
    )
    return parser.parse_args()


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_seed_from_name(path: Path) -> Optional[int]:
    match = re.search(r"_seed(\d+)\.csv$", path.name)
    if match:
        return int(match.group(1))
    return None


def read_csv_with_seed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "seed" not in df.columns:
        seed = extract_seed_from_name(path)
        if seed is not None:
            df["seed"] = seed
    return df


def get_summary_dirs(base_dir: Path, dataset: str) -> Dict[str, Path]:
    root = base_dir / "seed_summary"
    return {
        "root": root,
        "dataset_root": root / dataset,
        "csv": root / dataset / "csv",
        "figures": root / dataset / "figures",
        "logs": root / "logs",
    }


def write_log(log_path: Path, lines: Sequence[str]) -> None:
    safe_mkdir(log_path.parent)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def direction_consistency_from_counts(
    ad_count: int,
    control_count: int,
    neutral_count: int,
) -> float:
    total = ad_count + control_count + neutral_count
    if total <= 0:
        return float("nan")
    return max(ad_count, control_count, neutral_count) / total


def primary_direction_from_counts(
    ad_count: int,
    control_count: int,
    neutral_count: int,
) -> str:
    mapping = {
        "AD_support": ad_count,
        "Control_support": control_count,
        "Neutral": neutral_count,
    }
    return max(mapping.items(), key=lambda x: x[1])[0]


def standardize_window_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset

    rename_map = {}
    if "direction" in out.columns and "direction_window" not in out.columns:
        rename_map["direction"] = "direction_window"
    if "delta" in out.columns and "delta_window" not in out.columns:
        rename_map["delta"] = "delta_window"
    if "abs_delta" in out.columns and "abs_delta_window" not in out.columns:
        rename_map["abs_delta"] = "abs_delta_window"
    if rename_map:
        out = out.rename(columns=rename_map)

    for col in ["sample_id", "window_start", "window_end", "window_text"]:
        if col not in out.columns:
            out[col] = np.nan

    if "rank" not in out.columns:
        if "abs_delta_window" in out.columns:
            out["rank"] = (
                out.groupby("sample_id")["abs_delta_window"]
                .rank(method="dense", ascending=False)
                .astype(float)
            )
        else:
            out["rank"] = np.nan

    if "direction_window" not in out.columns:
        if "delta_window" in out.columns:
            out["direction_window"] = np.where(
                out["delta_window"] > 0,
                "AD_support",
                np.where(out["delta_window"] < 0, "Control_support", "Neutral"),
            )
        else:
            out["direction_window"] = "Neutral"

    if "abs_delta_window" not in out.columns and "delta_window" in out.columns:
        out["abs_delta_window"] = out["delta_window"].abs()

    out["sample_id"] = out["sample_id"].astype(str)
    out["window_text"] = out["window_text"].fillna("").astype(str)
    return out


def summarize_windows(
    df_all: pd.DataFrame,
    stable_threshold: float,
    min_seeds_for_stability: int,
) -> pd.DataFrame:
    key_cols = ["dataset", "sample_id", "window_start", "window_end", "window_text"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        n_seeds_found = int(g["seed"].nunique()) if "seed" in g.columns else len(g)
        ad_count = int((g["direction_window"] == "AD_support").sum())
        control_count = int((g["direction_window"] == "Control_support").sum())
        neutral_count = int((g["direction_window"] == "Neutral").sum())
        consistency = direction_consistency_from_counts(
            ad_count,
            control_count,
            neutral_count,
        )
        primary_direction = primary_direction_from_counts(
            ad_count,
            control_count,
            neutral_count,
        )

        return pd.Series(
            {
                "n_seeds_found": n_seeds_found,
                "mean_delta_window": (
                    float(g["delta_window"].mean())
                    if "delta_window" in g.columns
                    else np.nan
                ),
                "std_delta_window": (
                    float(g["delta_window"].std(ddof=0))
                    if "delta_window" in g.columns
                    else np.nan
                ),
                "mean_abs_delta_window": (
                    float(g["abs_delta_window"].mean())
                    if "abs_delta_window" in g.columns
                    else np.nan
                ),
                "std_abs_delta_window": (
                    float(g["abs_delta_window"].std(ddof=0))
                    if "abs_delta_window" in g.columns
                    else np.nan
                ),
                "min_delta_window": (
                    float(g["delta_window"].min())
                    if "delta_window" in g.columns
                    else np.nan
                ),
                "max_delta_window": (
                    float(g["delta_window"].max())
                    if "delta_window" in g.columns
                    else np.nan
                ),
                "mean_rank": float(g["rank"].mean()) if "rank" in g.columns else np.nan,
                "std_rank": (
                    float(g["rank"].std(ddof=0)) if "rank" in g.columns else np.nan
                ),
                "ad_support_count": ad_count,
                "control_support_count": control_count,
                "neutral_count": neutral_count,
                "primary_direction": primary_direction,
                "direction_consistency": consistency,
                "stable_window": (
                    bool(
                        consistency >= stable_threshold
                        and n_seeds_found >= min_seeds_for_stability
                    )
                    if pd.notna(consistency)
                    else False
                ),
            }
        )

    summary = df_all.groupby(key_cols, dropna=False).apply(_agg).reset_index()
    summary["stable_score"] = summary["direction_consistency"].fillna(0) * summary[
        "mean_abs_delta_window"
    ].fillna(0)
    summary = summary.sort_values(
        by=["stable_score", "mean_abs_delta_window", "direction_consistency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def detect_token_files(base_dir: Path, dataset: str) -> Tuple[List[Path], List[Path]]:
    token_dir = base_dir / "token_aggregation" / dataset / "csv"
    agg_files = sorted(token_dir.glob("token_attribution_aggregated_seed*.csv"))
    all_files = sorted(token_dir.glob("token_attribution_all_seed*.csv"))
    return agg_files, all_files


def standardize_token_columns(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset

    if "importance" not in out.columns:
        for cand in [
            "delta_token",
            "mean_importance",
            "raw_importance",
            "log_adjusted_importance",
            "token_importance",
            "score_change",
            "delta",
        ]:
            if cand in out.columns:
                out["importance"] = out[cand]
                break
    if "importance" not in out.columns:
        raise ValueError("No token importance-like column found.")

    if "abs_importance" not in out.columns:
        out["abs_importance"] = out["importance"].abs()

    if "direction_token" not in out.columns:
        out["direction_token"] = np.where(
            out["importance"] > 0,
            "AD_support",
            np.where(out["importance"] < 0, "Control_support", "Neutral"),
        )

    if "token" not in out.columns:
        for cand in ["token_text", "word"]:
            if cand in out.columns:
                out["token"] = out[cand]
                break
    if "token" not in out.columns:
        raise ValueError("No token column found.")

    if "sample_id" not in out.columns:
        out["sample_id"] = "GLOBAL"

    out["sample_id"] = out["sample_id"].astype(str)
    out["token"] = out["token"].fillna("").astype(str)
    return out


def summarize_tokens(
    df_all: pd.DataFrame,
    stable_threshold: float,
    min_seeds_for_stability: int,
) -> pd.DataFrame:
    key_cols = ["dataset", "sample_id", "token"]

    def _agg(g: pd.DataFrame) -> pd.Series:
        n_seeds_found = int(g["seed"].nunique()) if "seed" in g.columns else len(g)
        ad_count = int((g["direction_token"] == "AD_support").sum())
        control_count = int((g["direction_token"] == "Control_support").sum())
        neutral_count = int((g["direction_token"] == "Neutral").sum())
        consistency = direction_consistency_from_counts(
            ad_count,
            control_count,
            neutral_count,
        )
        primary_direction = primary_direction_from_counts(
            ad_count,
            control_count,
            neutral_count,
        )

        return pd.Series(
            {
                "n_seeds_found": n_seeds_found,
                "mean_importance": float(g["importance"].mean()),
                "std_importance": float(g["importance"].std(ddof=0)),
                "mean_abs_importance": float(g["abs_importance"].mean()),
                "std_abs_importance": float(g["abs_importance"].std(ddof=0)),
                "ad_support_count": ad_count,
                "control_support_count": control_count,
                "neutral_count": neutral_count,
                "primary_direction": primary_direction,
                "direction_consistency": consistency,
                "stable_token": (
                    bool(
                        consistency >= stable_threshold
                        and n_seeds_found >= min_seeds_for_stability
                    )
                    if pd.notna(consistency)
                    else False
                ),
            }
        )

    summary = df_all.groupby(key_cols, dropna=False).apply(_agg).reset_index()
    summary["stable_score"] = summary["direction_consistency"].fillna(0) * summary[
        "mean_abs_importance"
    ].fillna(0)
    summary = summary.sort_values(
        by=["stable_score", "mean_abs_importance", "direction_consistency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary


def compute_pairwise_seed_overlap_from_windows(
    df_all: pd.DataFrame,
    topk: int,
) -> pd.DataFrame:
    if "seed" not in df_all.columns:
        return pd.DataFrame()

    seeds = sorted(df_all["seed"].dropna().unique().tolist())
    if len(seeds) < 2:
        return pd.DataFrame()

    per_seed_top = {}
    for seed in seeds:
        sub = df_all[df_all["seed"] == seed].copy()
        if "rank" in sub.columns and sub["rank"].notna().any():
            top = (
                sub.sort_values(["sample_id", "rank"], ascending=[True, True])
                .groupby("sample_id")
                .head(topk)
            )
        else:
            top = (
                sub.sort_values(
                    ["sample_id", "abs_delta_window"],
                    ascending=[True, False],
                )
                .groupby("sample_id")
                .head(topk)
            )
        top["_window_key"] = (
            top["sample_id"].astype(str)
            + "||"
            + top["window_start"].astype(str)
            + "||"
            + top["window_end"].astype(str)
            + "||"
            + top["window_text"].astype(str)
        )
        per_seed_top[seed] = top

    rows = []
    for s1 in seeds:
        for s2 in seeds:
            if s1 == s2:
                overlap = 1.0
            else:
                a = per_seed_top[s1]
                b = per_seed_top[s2]
                sample_union = sorted(
                    set(a["sample_id"].unique()).union(set(b["sample_id"].unique()))
                )
                vals = []
                for sid in sample_union:
                    set_a = set(a.loc[a["sample_id"] == sid, "_window_key"].tolist())
                    set_b = set(b.loc[b["sample_id"] == sid, "_window_key"].tolist())
                    denom = max(1, min(len(set_a), len(set_b), topk))
                    vals.append(len(set_a.intersection(set_b)) / denom)
                overlap = float(np.mean(vals)) if vals else np.nan
            rows.append({"seed_1": s1, "seed_2": s2, "topk_overlap": overlap})
    return pd.DataFrame(rows)


def load_stability_files(base_dir: Path, dataset: str) -> pd.DataFrame:
    st_dir = base_dir / "debug_stability" / dataset / "csv"
    files = sorted(st_dir.glob("stability_summary*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_seed_overview(
    dataset: str,
    seed_list: Sequence[int],
    window_summary: pd.DataFrame,
    token_summary: Optional[pd.DataFrame],
    overlap_df: pd.DataFrame,
) -> pd.DataFrame:
    overlap_mean = np.nan
    overlap_std = np.nan
    if not overlap_df.empty and "topk_overlap" in overlap_df.columns:
        vals = overlap_df.loc[
            overlap_df["seed_1"] != overlap_df["seed_2"],
            "topk_overlap",
        ].dropna()
        if len(vals) > 0:
            overlap_mean = float(vals.mean())
            overlap_std = float(vals.std(ddof=0))

    row = {
        "dataset": dataset,
        "n_seed_files_detected": len(seed_list),
        "seed_list": ",".join(str(s) for s in seed_list),
        "n_unique_samples": (
            int(window_summary["sample_id"].nunique())
            if not window_summary.empty
            else 0
        ),
        "n_unique_windows": int(len(window_summary)),
        "n_stable_windows": (
            int(window_summary["stable_window"].sum())
            if "stable_window" in window_summary.columns
            else 0
        ),
        "mean_direction_consistency": (
            float(window_summary["direction_consistency"].mean())
            if "direction_consistency" in window_summary.columns and len(window_summary)
            else np.nan
        ),
        "mean_abs_delta_window": (
            float(window_summary["mean_abs_delta_window"].mean())
            if "mean_abs_delta_window" in window_summary.columns and len(window_summary)
            else np.nan
        ),
        "mean_std_delta_window": (
            float(window_summary["std_delta_window"].mean())
            if "std_delta_window" in window_summary.columns and len(window_summary)
            else np.nan
        ),
        "topk_overlap_mean": overlap_mean,
        "topk_overlap_std": overlap_std,
        "token_summary_available": bool(
            token_summary is not None and not token_summary.empty
        ),
    }
    return pd.DataFrame([row])


def maybe_shorten_text(text: str, max_chars: int = 60) -> str:
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def fig_window_delta_by_seed(df_all: pd.DataFrame, out_path: Path) -> None:
    seeds = sorted(df_all["seed"].dropna().unique().tolist())
    data = [
        df_all.loc[df_all["seed"] == s, "delta_window"].dropna().values for s in seeds
    ]
    if not data:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[str(s) for s in seeds], vert=True)
    plt.xlabel("Seed")
    plt.ylabel("delta_window")
    plt.title("Window delta distribution by seed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_window_abs_delta_by_seed(df_all: pd.DataFrame, out_path: Path) -> None:
    seeds = sorted(df_all["seed"].dropna().unique().tolist())
    data = [
        df_all.loc[df_all["seed"] == s, "abs_delta_window"].dropna().values
        for s in seeds
    ]
    if not data:
        return
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=[str(s) for s in seeds], vert=True)
    plt.xlabel("Seed")
    plt.ylabel("abs_delta_window")
    plt.title("Absolute window delta distribution by seed")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_direction_consistency_hist(summary_df: pd.DataFrame, out_path: Path) -> None:
    vals = summary_df["direction_consistency"].dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=20)
    plt.xlabel("direction_consistency")
    plt.ylabel("Count")
    plt.title("Window direction consistency")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_delta_std_hist(summary_df: pd.DataFrame, out_path: Path) -> None:
    vals = summary_df["std_delta_window"].dropna().values
    if len(vals) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=20)
    plt.xlabel("std_delta_window")
    plt.ylabel("Count")
    plt.title("Window delta standard deviation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_top_stable_windows(
    summary_df: pd.DataFrame,
    out_path: Path,
    topk: int,
) -> None:
    if summary_df.empty:
        return
    top = summary_df.head(topk).copy()
    labels = [
        f"{sid} | {maybe_shorten_text(txt, 45)}"
        for sid, txt in zip(top["sample_id"], top["window_text"])
    ]
    values = top["stable_score"].fillna(0).tolist()

    plt.figure(figsize=(12, max(6, 0.4 * len(top))))
    y = np.arange(len(top))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel("stable_score")
    plt.title(f"Top {len(top)} stable windows")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fig_seed_overlap_heatmap(overlap_df: pd.DataFrame, out_path: Path) -> None:
    if overlap_df.empty:
        return
    # Some sources (e.g. per-sample stability summaries) contain duplicated
    # seed-pair rows; aggregate first so pivot always has unique index/columns.
    overlap_df = (
        overlap_df.groupby(["seed_1", "seed_2"], as_index=False)["topk_overlap"]
        .mean()
        .copy()
    )
    pivot = (
        overlap_df.pivot(index="seed_1", columns="seed_2", values="topk_overlap")
        .sort_index()
        .sort_index(axis=1)
    )
    arr = pivot.values

    plt.figure(figsize=(8, 6))
    im = plt.imshow(arr, aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel("Seed")
    plt.ylabel("Seed")
    plt.title("Seed pairwise top-k overlap")
    cbar = plt.colorbar(im)
    cbar.set_label("topk_overlap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def export_summary_csvs(
    out_dir: Path,
    window_summary: pd.DataFrame,
    topk: int,
    overview_df: pd.DataFrame,
    token_summary: Optional[pd.DataFrame],
) -> None:
    safe_mkdir(out_dir)
    window_summary.to_csv(out_dir / "window_summary_across_seeds.csv", index=False)
    overview_df.to_csv(out_dir / "seed_overview.csv", index=False)
    window_summary.head(topk).to_csv(out_dir / "top_stable_windows.csv", index=False)

    if token_summary is not None and not token_summary.empty:
        token_summary.to_csv(out_dir / "token_summary_across_seeds.csv", index=False)
        token_summary.head(topk).to_csv(out_dir / "top_stable_tokens.csv", index=False)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    log_lines = []

    for dataset in datasets:
        dirs = get_summary_dirs(base_dir, dataset)
        window_dir = base_dir / "window_extraction" / dataset / "csv"
        window_files = sorted(window_dir.glob("window_importance_seed*.csv"))

        seed_list = sorted(
            {
                extract_seed_from_name(f)
                for f in window_files
                if extract_seed_from_name(f) is not None
            }
        )
        if len(seed_list) < args.min_seeds:
            log_lines.append(
                f"[SKIP] dataset={dataset}: found {len(seed_list)} seed-specific "
                f"window files; need at least {args.min_seeds}."
            )
            continue

        safe_mkdir(dirs["csv"])
        safe_mkdir(dirs["figures"])
        safe_mkdir(dirs["logs"])

        window_frames = []
        for f in window_files:
            try:
                df = read_csv_with_seed(f)
                df = standardize_window_columns(df, dataset)
                window_frames.append(df)
            except Exception as e:
                log_lines.append(
                    f"[WARN] dataset={dataset}: failed to read {f.name}: {e}"
                )

        if not window_frames:
            log_lines.append(
                f"[SKIP] dataset={dataset}: no readable window files after scanning."
            )
            continue

        window_all = pd.concat(window_frames, ignore_index=True)
        if (
            "seed" not in window_all.columns
            or window_all["seed"].dropna().nunique() < args.min_seeds
        ):
            log_lines.append(
                f"[SKIP] dataset={dataset}: distinct seeds in loaded window files "
                f"are insufficient."
            )
            continue

        window_summary = summarize_windows(
            window_all,
            stable_threshold=args.stable_threshold,
            min_seeds_for_stability=args.min_seeds,
        )

        token_summary = None
        token_agg_files, token_all_files = detect_token_files(base_dir, dataset)
        chosen_token_files = token_agg_files if token_agg_files else token_all_files
        token_frames = []

        if chosen_token_files:
            for f in chosen_token_files:
                try:
                    tdf = read_csv_with_seed(f)
                    tdf = standardize_token_columns(tdf, dataset)
                    token_frames.append(tdf)
                except Exception as e:
                    log_lines.append(
                        f"[WARN] dataset={dataset}: failed to parse token file "
                        f"{f.name}: {e}"
                    )

            if token_frames:
                try:
                    token_all = pd.concat(token_frames, ignore_index=True)
                    if (
                        "seed" in token_all.columns
                        and token_all["seed"].dropna().nunique() >= args.min_seeds
                    ):
                        token_summary = summarize_tokens(
                            token_all,
                            stable_threshold=args.stable_threshold,
                            min_seeds_for_stability=args.min_seeds,
                        )
                    else:
                        log_lines.append(
                            f"[INFO] dataset={dataset}: token files found but seed "
                            f"count is insufficient for token summary."
                        )
                except Exception as e:
                    log_lines.append(
                        f"[WARN] dataset={dataset}: token summary failed: {e}"
                    )
        else:
            log_lines.append(
                f"[INFO] dataset={dataset}: no token seed files found; token summary skipped."
            )

        stability_df = load_stability_files(base_dir, dataset)
        if not stability_df.empty and {
            "reference_seed",
            "compare_seed",
            "topk_overlap",
        }.issubset(stability_df.columns):
            overlap_df = (
                stability_df[["reference_seed", "compare_seed", "topk_overlap"]]
                .rename(
                    columns={
                        "reference_seed": "seed_1",
                        "compare_seed": "seed_2",
                    }
                )
                .copy()
            )
        else:
            overlap_df = compute_pairwise_seed_overlap_from_windows(
                window_all,
                topk=args.topk,
            )

        overview_df = make_seed_overview(
            dataset=dataset,
            seed_list=sorted(window_all["seed"].dropna().unique().tolist()),
            window_summary=window_summary,
            token_summary=token_summary,
            overlap_df=overlap_df,
        )

        export_summary_csvs(
            dirs["csv"],
            window_summary,
            args.topk,
            overview_df,
            token_summary,
        )

        try:
            fig_window_delta_by_seed(
                window_all,
                dirs["figures"] / "window_delta_distribution_by_seed.png",
            )
            fig_window_abs_delta_by_seed(
                window_all,
                dirs["figures"] / "window_abs_delta_distribution_by_seed.png",
            )
            fig_direction_consistency_hist(
                window_summary,
                dirs["figures"] / "window_direction_consistency_hist.png",
            )
            fig_delta_std_hist(
                window_summary,
                dirs["figures"] / "window_delta_std_hist.png",
            )
            fig_top_stable_windows(
                window_summary,
                dirs["figures"] / "top_stable_windows.png",
                args.topk,
            )
            if not overlap_df.empty:
                fig_seed_overlap_heatmap(
                    overlap_df,
                    dirs["figures"] / "seed_overlap_heatmap.png",
                )
        except Exception as e:
            log_lines.append(
                f"[WARN] dataset={dataset}: figure generation failed partially: {e}"
            )

        log_lines.append(
            f"[DONE] dataset={dataset}: "
            f"seeds={sorted(window_all['seed'].dropna().unique().tolist())}, "
            f"windows={len(window_summary)}, "
            f"stable_windows={int(window_summary['stable_window'].sum())}, "
            f"token_summary={'yes' if token_summary is not None and not token_summary.empty else 'no'}"
        )

    log_path = base_dir / "seed_summary" / "logs" / "seed_summary_output.txt"
    write_log(log_path, log_lines)
    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
