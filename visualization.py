"""
Visualization module for the AD attribution experiment.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    DATASET_PATHS,
    build_dataset_dirs,
    ensure_dirs,
)

warnings.filterwarnings("ignore")


def add_seed_suffix(path, seed):
    """Append seed suffix before file extension."""
    base, ext = os.path.splitext(str(path))
    return f"{base}_seed{seed}{ext}"


class ResultsVisualizer:
    def __init__(self, output_dir, input_paths=None, dataset_name="test"):
        self.output_dir = str(output_dir)
        self.dataset_name = dataset_name
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.input_paths = input_paths or {}
        self.window_df = None
        self.token_df = None
        self.aggregated_df = None
        self.stability_df = None

        self.load_data()

        plt.rcParams["figure.figsize"] = (12, 7)
        plt.rcParams["font.size"] = 10

    def load_data(self):
        for attr, path in self.input_paths.items():
            if os.path.exists(path):
                setattr(self, attr, pd.read_csv(path))
                print(f"Loaded {attr}: {path}")
            else:
                print(f"Warning: missing input file for {attr}: {path}")

    def validate_required_inputs(self):
        required = {
            "window_df": "window extraction output",
            "token_df": "token attribution all output",
            "aggregated_df": "token attribution aggregated output",
        }

        missing = []
        for attr, desc in required.items():
            if getattr(self, attr) is None:
                missing.append(f"{attr} ({desc})")

        if missing:
            print("Warning: some required visualization inputs are missing:")
            for item in missing:
                print(f"  - {item}")

    def available_token_metrics(self):
        if self.aggregated_df is None or self.aggregated_df.empty:
            return []

        preferred_order = [
            "raw_importance",
            "mean_importance",
            "log_adjusted_importance",
            "signed_mean_delta",
        ]
        return [m for m in preferred_order if m in self.aggregated_df.columns]

    def top_tokens_for_analysis(self, top_n=20, metric="log_adjusted_importance"):
        if self.aggregated_df is None or self.aggregated_df.empty:
            return pd.DataFrame()

        usable_metric = metric
        if usable_metric not in self.aggregated_df.columns:
            fallback_order = [
                "log_adjusted_importance",
                "raw_importance",
                "mean_importance",
            ]
            usable_metric = next(
                (m for m in fallback_order if m in self.aggregated_df.columns),
                None,
            )

        if usable_metric is None:
            return pd.DataFrame()

        return self.aggregated_df.nlargest(top_n, usable_metric).copy()

    def plot_window_direction_counts(self):
        if self.window_df is None or self.window_df.empty:
            return
        if "direction_window" not in self.window_df.columns:
            return

        plot_df = self.window_df["direction_window"].value_counts().sort_index()

        plt.figure()
        plt.bar(plot_df.index.astype(str), plot_df.values)
        plt.title(f"Window Direction Counts ({self.dataset_name})")
        plt.xlabel("Direction")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "window_direction_counts.png"), dpi=300
        )
        plt.close()

    def plot_top_windows(self):
        if self.window_df is None or self.window_df.empty:
            return
        if "abs_delta_window" not in self.window_df.columns:
            return

        plot_df = self.window_df.nlargest(20, "abs_delta_window").copy()
        labels = (
            plot_df["sample_id"].astype(str)
            + ": "
            + plot_df["window_text"].astype(str)
            + " | delta="
            + plot_df["delta_window"].map(lambda x: f"{x:.4f}")
            + " | abs="
            + plot_df["abs_delta_window"].map(lambda x: f"{x:.4f}")
        ).tolist()

        plt.figure(figsize=(14, 8))
        plt.barh(range(len(plot_df)), plot_df["abs_delta_window"].values)
        plt.yticks(range(len(plot_df)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(f"Top Windows by Absolute Delta ({self.dataset_name})")
        plt.xlabel("abs_delta_window")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "top_windows_abs_delta.png"), dpi=300)
        plt.close()

    def plot_directional_windows(self, direction, filename):
        if (
            self.window_df is None
            or self.window_df.empty
            or "direction_window" not in self.window_df.columns
            or "delta_window" not in self.window_df.columns
            or "abs_delta_window" not in self.window_df.columns
        ):
            return

        plot_df = self.window_df[self.window_df["direction_window"] == direction].copy()
        if plot_df.empty:
            return

        if direction == "AD_support":
            title = f"Top AD-Support Windows by Absolute Delta ({self.dataset_name})"
        else:
            title = (
                f"Top Control-Support Windows by Absolute Delta ({self.dataset_name})"
            )

        plot_df = plot_df.nlargest(20, "abs_delta_window")
        labels = (
            plot_df["sample_id"].astype(str)
            + ": "
            + plot_df["window_text"].astype(str)
            + " | delta="
            + plot_df["delta_window"].map(lambda x: f"{x:.4f}")
            + " | abs="
            + plot_df["abs_delta_window"].map(lambda x: f"{x:.4f}")
        ).tolist()

        plt.figure(figsize=(14, 8))
        plt.barh(range(len(plot_df)), plot_df["abs_delta_window"].values)
        plt.yticks(range(len(plot_df)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("abs_delta_window")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_top_tokens(self, metric):
        if self.aggregated_df is None or self.aggregated_df.empty:
            return
        if metric not in self.aggregated_df.columns:
            return
        if "token" not in self.aggregated_df.columns:
            return

        plot_df = self.aggregated_df.nlargest(20, metric).copy()

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(plot_df)), plot_df[metric].values)
        plt.yticks(
            range(len(plot_df)), plot_df["token"].astype(str).tolist(), fontsize=9
        )
        plt.gca().invert_yaxis()
        plt.title(f"Top Tokens by {metric} ({self.dataset_name})")
        plt.xlabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"top_tokens_{metric}.png"), dpi=300)
        plt.close()

    def plot_top_occurrences(self):
        if (
            self.token_df is None
            or self.token_df.empty
            or "occurrence_id" not in self.token_df.columns
            or "delta_token" not in self.token_df.columns
            or "abs_delta_token" not in self.token_df.columns
        ):
            return

        plot_df = (
            self.token_df.drop_duplicates(subset=["occurrence_id"])
            .nlargest(20, "abs_delta_token")
            .copy()
        )

        labels = (
            plot_df["occurrence_id"].astype(str)
            + ": "
            + plot_df["token"].astype(str)
            + " | delta="
            + plot_df["delta_token"].map(lambda x: f"{x:.4f}")
            + " | abs="
            + plot_df["abs_delta_token"].map(lambda x: f"{x:.4f}")
        ).tolist()

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(plot_df)), plot_df["abs_delta_token"].values)
        plt.yticks(range(len(plot_df)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(f"Top Token Occurrences by Absolute Delta ({self.dataset_name})")
        plt.xlabel("abs_delta_token")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "top_occurrences_abs_delta.png"), dpi=300
        )
        plt.close()

    def plot_directional_occurrences(self, direction, filename):
        if (
            self.token_df is None
            or self.token_df.empty
            or "occurrence_id" not in self.token_df.columns
            or "direction_token" not in self.token_df.columns
            or "delta_token" not in self.token_df.columns
            or "abs_delta_token" not in self.token_df.columns
        ):
            return

        occurrence_df = self.token_df.drop_duplicates(subset=["occurrence_id"]).copy()
        plot_df = occurrence_df[occurrence_df["direction_token"] == direction].copy()

        if plot_df.empty:
            return

        if direction == "AD_support":
            title = f"Top AD-Support Token Occurrences by Absolute Delta ({self.dataset_name})"
        else:
            title = f"Top Control-Support Token Occurrences by Absolute Delta ({self.dataset_name})"

        plot_df = plot_df.nlargest(20, "abs_delta_token")
        labels = (
            plot_df["occurrence_id"].astype(str)
            + ": "
            + plot_df["token"].astype(str)
            + " | delta="
            + plot_df["delta_token"].map(lambda x: f"{x:.4f}")
            + " | abs="
            + plot_df["abs_delta_token"].map(lambda x: f"{x:.4f}")
        ).tolist()

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(plot_df)), plot_df["abs_delta_token"].values)
        plt.yticks(range(len(plot_df)), labels, fontsize=8)
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("abs_delta_token")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_top20_token_direction_distribution(self, metric="log_adjusted_importance"):
        """
        Stacked bar chart for top 20 tokens:
        AD_support_count / Control_support_count / Neutral_count
        """
        if self.aggregated_df is None or self.aggregated_df.empty:
            return

        required = {
            "token",
            "ad_support_count",
            "control_support_count",
            "neutral_count",
        }
        if not required.issubset(self.aggregated_df.columns):
            return

        plot_df = self.top_tokens_for_analysis(top_n=20, metric=metric)
        if plot_df.empty:
            return

        x = np.arange(len(plot_df))
        ad_vals = plot_df["ad_support_count"].to_numpy()
        control_vals = plot_df["control_support_count"].to_numpy()
        neutral_vals = plot_df["neutral_count"].to_numpy()

        plt.figure(figsize=(14, 8))
        plt.bar(x, ad_vals, label="AD_support")
        plt.bar(x, control_vals, bottom=ad_vals, label="Control_support")
        plt.bar(x, neutral_vals, bottom=ad_vals + control_vals, label="Neutral")
        plt.xticks(x, plot_df["token"].astype(str), rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title(f"Top 20 Token Direction Distribution ({self.dataset_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "top20_token_direction_distribution.png"),
            dpi=300,
        )
        plt.close()

    def plot_top20_signed_mean_delta(self, metric="log_adjusted_importance"):
        """
        Bar chart of signed_mean_delta for top 20 tokens.
        """
        if self.aggregated_df is None or self.aggregated_df.empty:
            return
        required = {"token", "signed_mean_delta"}
        if not required.issubset(self.aggregated_df.columns):
            return

        plot_df = self.top_tokens_for_analysis(top_n=20, metric=metric)
        if plot_df.empty:
            return

        plt.figure(figsize=(14, 8))
        plt.bar(plot_df["token"].astype(str), plot_df["signed_mean_delta"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("signed_mean_delta")
        plt.title(f"Top 20 Tokens by Signed Mean Delta ({self.dataset_name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "top20_signed_mean_delta.png"),
            dpi=300,
        )
        plt.close()

    def plot_top20_token_delta_boxplot(self, metric="log_adjusted_importance"):
        """
        Boxplot of occurrence-level delta_token for top 20 tokens.
        Useful for seeing dispersion / bimodality / instability.
        """
        if self.token_df is None or self.token_df.empty:
            return
        if self.aggregated_df is None or self.aggregated_df.empty:
            return

        required_token = {"token", "delta_token"}
        required_agg = {"token"}
        if not required_token.issubset(self.token_df.columns):
            return
        if not required_agg.issubset(self.aggregated_df.columns):
            return

        top_df = self.top_tokens_for_analysis(top_n=20, metric=metric)
        if top_df.empty:
            return

        top_tokens = top_df["token"].astype(str).tolist()
        plot_df = self.token_df[
            self.token_df["token"].astype(str).isin(top_tokens)
        ].copy()
        if plot_df.empty:
            return

        grouped_data = []
        labels = []
        for token in top_tokens:
            vals = plot_df.loc[
                plot_df["token"].astype(str) == token, "delta_token"
            ].dropna()
            if len(vals) > 0:
                grouped_data.append(vals.to_numpy())
                labels.append(token)

        if not grouped_data:
            return

        plt.figure(figsize=(14, 8))
        plt.boxplot(grouped_data, labels=labels, vert=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("delta_token")
        plt.title(f"Top 20 Token Delta Distribution ({self.dataset_name})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "top20_token_delta_boxplot.png"),
            dpi=300,
        )
        plt.close()

    def plot_token_sensitivity_stability_quadrants(
        self,
        sensitivity_metric="log_adjusted_importance",
    ):
        """
        Quadrant plot:
            x-axis = token stability
            y-axis = token sensitivity

        stability = max(ad_support_ratio, 1 - ad_support_ratio)
        sensitivity = log_adjusted_importance (preferred)
        """
        if self.aggregated_df is None or self.aggregated_df.empty:
            return

        required = {"token", "ad_support_ratio"}
        if not required.issubset(self.aggregated_df.columns):
            return

        sensitivity_col = sensitivity_metric
        if sensitivity_col not in self.aggregated_df.columns:
            fallback_order = [
                "log_adjusted_importance",
                "raw_importance",
                "mean_importance",
            ]
            sensitivity_col = next(
                (m for m in fallback_order if m in self.aggregated_df.columns),
                None,
            )

        if sensitivity_col is None:
            return

        plot_df = self.aggregated_df.copy()
        plot_df["stability_score"] = np.maximum(
            plot_df["ad_support_ratio"],
            1.0 - plot_df["ad_support_ratio"],
        )
        plot_df["sensitivity_score"] = plot_df[sensitivity_col]

        x = plot_df["stability_score"].to_numpy()
        y = plot_df["sensitivity_score"].to_numpy()

        x_cut = float(np.median(x))
        y_cut = float(np.median(y))

        plt.figure(figsize=(10, 8))
        plt.scatter(x, y, alpha=0.7)

        plt.axvline(x=x_cut, linestyle="--")
        plt.axhline(y=y_cut, linestyle="--")

        plt.xlabel("Token Stability")
        plt.ylabel(f"Token Sensitivity ({sensitivity_col})")
        plt.title(f"Token Sensitivity × Stability Quadrants ({self.dataset_name})")

        # annotate top tokens in Q1 or globally strongest few
        annotate_df = plot_df.nlargest(min(15, len(plot_df)), "sensitivity_score")
        for _, row in annotate_df.iterrows():
            plt.annotate(
                str(row["token"]),
                (row["stability_score"], row["sensitivity_score"]),
                fontsize=8,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.output_dir,
                "token_sensitivity_stability_quadrants.png",
            ),
            dpi=300,
        )
        plt.close()

        # optional export of quadrant labels
        plot_df["quadrant"] = np.select(
            [
                (plot_df["stability_score"] >= x_cut)
                & (plot_df["sensitivity_score"] >= y_cut),
                (plot_df["stability_score"] < x_cut)
                & (plot_df["sensitivity_score"] >= y_cut),
                (plot_df["stability_score"] < x_cut)
                & (plot_df["sensitivity_score"] < y_cut),
                (plot_df["stability_score"] >= x_cut)
                & (plot_df["sensitivity_score"] < y_cut),
            ],
            [
                "Q1_high_sensitivity_high_stability",
                "Q2_high_sensitivity_low_stability",
                "Q3_low_sensitivity_low_stability",
                "Q4_low_sensitivity_high_stability",
            ],
            default="unknown",
        )

        quadrant_csv = os.path.join(
            self.output_dir,
            "token_quadrant_assignment.csv",
        )
        plot_df.to_csv(quadrant_csv, index=False)

    def plot_stability_summary(self):
        if self.stability_df is None or self.stability_df.empty:
            return

        if "topk_overlap" in self.stability_df.columns:
            plot_df = (
                self.stability_df.groupby("compare_seed", as_index=False)[
                    "topk_overlap"
                ]
                .mean()
                .copy()
            )

            plt.figure(figsize=(10, 6))
            plt.bar(plot_df["compare_seed"].astype(str), plot_df["topk_overlap"])
            plt.ylim(0, 1.05)
            plt.title(f"Stability Summary ({self.dataset_name})")
            plt.xlabel("Compare Seed")
            plt.ylabel("Mean Top-k Overlap")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "stability_summary.png"), dpi=300)
            plt.close()
            return

        required = {"reference_seed", "compare_seed", "metric", "mean_value"}
        if not required.issubset(set(self.stability_df.columns)):
            print(
                "Warning: stability_df exists but does not contain expected columns for plotting."
            )
            return

        plot_df = self.stability_df.copy()
        plot_df["seed_pair"] = (
            plot_df["reference_seed"].astype(str)
            + "->"
            + plot_df["compare_seed"].astype(str)
        )

        pivot = plot_df.pivot_table(
            index="metric",
            columns="seed_pair",
            values="mean_value",
            aggfunc="mean",
        ).fillna(0.0)

        plt.figure(figsize=(14, 8))
        for seed_pair in pivot.columns:
            plt.plot(pivot.index, pivot[seed_pair], marker="o", label=str(seed_pair))

        plt.ylim(0, 1.05)
        plt.title(f"Stability Summary Metrics ({self.dataset_name})")
        plt.xlabel("Metric")
        plt.ylabel("Mean value")
        plt.xticks(rotation=20, ha="right")
        plt.legend(title="Seed pair")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "stability_summary.png"), dpi=300)
        plt.close()

    def generate_all(self):
        self.validate_required_inputs()

        self.plot_window_direction_counts()
        self.plot_top_windows()
        self.plot_directional_windows("AD_support", "top_windows_ad_support.png")
        self.plot_directional_windows(
            "Control_support", "top_windows_control_support.png"
        )
        self.plot_top_occurrences()
        self.plot_directional_occurrences(
            "AD_support", "top_occurrences_ad_support.png"
        )
        self.plot_directional_occurrences(
            "Control_support", "top_occurrences_control_support.png"
        )

        metrics = self.available_token_metrics()
        if not metrics:
            print("Warning: no usable token metrics found in aggregated_df.")
        else:
            print(f"Token metrics to plot: {metrics}")
            for metric in metrics:
                self.plot_top_tokens(metric)

        # newly added interpretation-oriented plots
        self.plot_top20_token_direction_distribution()
        self.plot_top20_signed_mean_delta()
        self.plot_top20_token_delta_boxplot()
        self.plot_token_sensitivity_stability_quadrants()

        self.plot_stability_summary()
        print(f"Figures saved to {self.output_dir}")


def main(seed=None, output_with_seed=False, dataset_name="test", dataset_path=None):
    vis_dirs = build_dataset_dirs("visualization", dataset_name)
    window_dirs = build_dataset_dirs("window_extraction", dataset_name)
    token_dirs = build_dataset_dirs("token_aggregation", dataset_name)
    stability_dirs = build_dataset_dirs("debug_stability", dataset_name)
    ensure_dirs(vis_dirs, keys=["figures"])

    output_dir = vis_dirs["figures"]

    input_paths = {
        "window_df": str(Path(window_dirs["csv"]) / "window_importance.csv"),
        "token_df": str(Path(token_dirs["csv"]) / "token_attribution_all.csv"),
        "aggregated_df": str(
            Path(token_dirs["csv"]) / "token_attribution_aggregated.csv"
        ),
        "stability_df": str(Path(stability_dirs["csv"]) / "stability_summary.csv"),
    }

    if output_with_seed and seed is not None:
        output_dir = str(Path(vis_dirs["figures"]) / f"seed_{seed}")
        input_paths = {
            "window_df": add_seed_suffix(input_paths["window_df"], seed),
            "token_df": add_seed_suffix(input_paths["token_df"], seed),
            "aggregated_df": add_seed_suffix(input_paths["aggregated_df"], seed),
            "stability_df": add_seed_suffix(input_paths["stability_df"], seed),
        }

    visualizer = ResultsVisualizer(
        output_dir=output_dir,
        input_paths=input_paths,
        dataset_name=dataset_name,
    )
    visualizer.generate_all()


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASET_PATHS.items():
        main(
            seed=None,
            output_with_seed=False,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
