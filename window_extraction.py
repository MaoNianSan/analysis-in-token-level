import os
import warnings
import io
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist, pdist

from config import (
    RANDOM_SEED,
    WINDOW_LENGTH,
    STRIDE,
    N_SAMPLES,
    NOISE_SCALE,
    VECTORIZER_PATH,
    MODEL_PATH,
    DATASET_PATHS,
    build_dataset_dirs,
    ensure_dirs,
    set_global_seed,
    compute_direction,
)

warnings.filterwarnings("ignore")


def add_seed_suffix(path, seed):
    """Append seed suffix before file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}_seed{seed}{ext}"


class TeeOutput:
    """Write stdout to both the console and an in-memory buffer."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_occurrence_id(sample_id, token_position):
    return f"{sample_id}:{token_position}"


def get_windows(text, sample_id=None, window_length=WINDOW_LENGTH, stride=STRIDE):
    """Return sliding windows over whitespace tokens."""
    tokens = str(text).split()
    windows = []

    for start in range(0, len(tokens) - window_length + 1, stride):
        end = start + window_length
        token_positions = list(range(start, end))
        occurrence_ids = (
            [make_occurrence_id(sample_id, pos) for pos in token_positions]
            if sample_id is not None
            else []
        )

        windows.append(
            {
                "window_text": " ".join(tokens[start:end]),
                "window_start": start,
                "window_end": end,
                "window_token_positions": token_positions,
                "window_occurrence_ids": occurrence_ids,
            }
        )
    return windows


def remove_window(tokens, start_idx, end_idx):
    """Delete a window from the token sequence."""
    return " ".join(tokens[:start_idx] + tokens[end_idx:])


def get_ad_score(text, vectorizer, clf):
    """Deterministic AD probability for one text."""
    if not str(text).strip():
        return 0.5
    features = vectorizer.transform([str(text)])
    return float(clf.predict_proba(features)[0, 1])


def get_ad_score_distribution(
    text, vectorizer, clf, seed, n_samples=N_SAMPLES, noise_scale=NOISE_SCALE
):
    """
    Seeded noisy score distribution used only as a supplemental stability column.
    Primary interpretation uses delta_window / abs_delta_window.
    """
    if not str(text).strip():
        return [0.5] * n_samples

    rng = np.random.default_rng(seed)
    X_dense = vectorizer.transform([str(text)]).toarray()
    scores = []

    for _ in range(n_samples):
        noisy = X_dense + rng.normal(0, noise_scale, X_dense.shape)
        score = float(clf.predict_proba(noisy)[0, 1])
        scores.append(score)

    return scores


def compute_energy_distance(dist1, dist2):
    """Supplemental distribution distance; not used for primary ranking."""
    d1 = np.asarray(dist1, dtype=float).reshape(-1, 1)
    d2 = np.asarray(dist2, dtype=float).reshape(-1, 1)

    d1_intra = np.mean(pdist(d1)) if len(d1) > 1 else 0.0
    d2_intra = np.mean(pdist(d2)) if len(d2) > 1 else 0.0
    d_inter = np.mean(cdist(d1, d2))

    return float(max(0.0, 2 * d_inter - d1_intra - d2_intra))


def resolve_dataset_path(dataset_name, dataset_path):
    """Resolve dataset path with dataset_name priority if path not supplied."""
    if dataset_path is not None:
        return dataset_path
    if dataset_name not in DATASET_PATHS:
        raise ValueError(
            f"Unknown dataset_name='{dataset_name}'. "
            f"Expected one of {list(DATASET_PATHS.keys())}."
        )
    return DATASET_PATHS[dataset_name]


def main(
    seed=RANDOM_SEED,
    output_with_seed=False,
    dataset_name="test",
    dataset_path=None,
):
    set_global_seed(seed)

    dataset_path = resolve_dataset_path(dataset_name, dataset_path)
    dataset_dirs = build_dataset_dirs("window_extraction", dataset_name)
    ensure_dirs(dataset_dirs, keys=["csv", "logs"])

    vectorizer_path = VECTORIZER_PATH
    model_path = MODEL_PATH

    window_importance_path = str(Path(dataset_dirs["csv"]) / "window_importance.csv")
    window_extraction_output_path = str(
        Path(dataset_dirs["logs"]) / "window_extraction_output.txt"
    )

    if output_with_seed:
        window_importance_path = add_seed_suffix(window_importance_path, seed)
        window_extraction_output_path = add_seed_suffix(
            window_extraction_output_path, seed
        )

    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = TeeOutput(old_stdout, captured_output)

    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)

    df = pd.read_csv(dataset_path)
    df["Speech"] = df["Speech"].fillna("")

    all_results = []

    print("=" * 80)
    print(f"Window Sliding Analysis | dataset={dataset_name} | seed={seed}")
    print("=" * 80)
    print(f"Dataset path: {dataset_path}")
    print(f"Vectorizer path: {vectorizer_path}")
    print(f"Model path: {model_path}")
    print(f"Window length={WINDOW_LENGTH}, stride={STRIDE}")
    print(f"Distribution samples={N_SAMPLES}, noise_scale={NOISE_SCALE}")

    for idx, row in df.iterrows():
        text = str(row.get("Speech", "")).strip()
        if not text:
            continue

        tokens = text.split()
        if len(tokens) < WINDOW_LENGTH:
            continue

        sample_id = row.get("sample_id", idx)
        true_label = row.get("label")
        original_prob = get_ad_score(text, vectorizer, clf)

        features = vectorizer.transform([text])
        predicted_label = clf.predict(features)[0]
        correct = (predicted_label == true_label) if pd.notna(true_label) else None

        original_distribution = get_ad_score_distribution(
            text,
            vectorizer,
            clf,
            seed=seed * 100000 + idx,
        )

        sample_windows = []
        windows = get_windows(text, sample_id=sample_id)

        for window_idx, window in enumerate(windows):
            text_without_window = remove_window(
                tokens, window["window_start"], window["window_end"]
            )

            score_without_window = get_ad_score(text_without_window, vectorizer, clf)

            distribution_without_window = get_ad_score_distribution(
                text_without_window,
                vectorizer,
                clf,
                seed=seed * 100000000 + idx * 1000 + window_idx + 1,
            )

            delta_window = original_prob - score_without_window
            abs_delta_window = abs(delta_window)
            direction_window = compute_direction(delta_window)

            energy_distance = compute_energy_distance(
                original_distribution,
                distribution_without_window,
            )

            sample_windows.append(
                {
                    "seed": seed,
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "correct": correct,
                    "original_prob": original_prob,
                    "window_start": window["window_start"],
                    "window_end": window["window_end"],
                    "window_text": window["window_text"],
                    "window_token_positions": window["window_token_positions"],
                    "window_occurrence_ids": window["window_occurrence_ids"],
                    "score_without_window": score_without_window,
                    "delta_window": delta_window,
                    "abs_delta_window": abs_delta_window,
                    "direction_window": direction_window,
                    "energy_distance": energy_distance,
                }
            )

        sample_windows = sorted(
            sample_windows,
            key=lambda x: (x["abs_delta_window"], x["energy_distance"]),
            reverse=True,
        )

        for rank, record in enumerate(sample_windows, start=1):
            record["rank"] = rank
            all_results.append(record)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} samples...")

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        results_df["window_token_positions"] = results_df[
            "window_token_positions"
        ].apply(json.dumps)
        results_df["window_occurrence_ids"] = results_df["window_occurrence_ids"].apply(
            json.dumps
        )

    results_df.to_csv(window_importance_path, index=False)

    print("\nSaved files:")
    print(f"- {window_importance_path} ({len(results_df)} windows)")
    print(f"- {window_extraction_output_path}")

    if not results_df.empty:
        print("\nTop 10 windows across all samples:")
        preview = results_df.nlargest(10, "abs_delta_window")[
            [
                "seed",
                "dataset",
                "sample_id",
                "window_text",
                "delta_window",
                "abs_delta_window",
                "direction_window",
                "score_without_window",
            ]
        ]

        for _, row in preview.iterrows():
            print(
                f"Seed {row['seed']} | Dataset {row['dataset']} | Sample {row['sample_id']}: "
                f"'{row['window_text']}' | "
                f"delta={row['delta_window']:.4f}, "
                f"abs_delta={row['abs_delta_window']:.4f}, "
                f"direction={row['direction_window']}"
            )

    print("\n" + "=" * 80)

    sys.stdout = old_stdout
    output_content = captured_output.getvalue()

    with open(window_extraction_output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    return {
        "seed": seed,
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "vectorizer_path": vectorizer_path,
        "model_path": model_path,
        "window_importance_path": window_importance_path,
        "window_extraction_output_path": window_extraction_output_path,
        "n_windows": len(results_df),
    }


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASET_PATHS.items():
        main(
            seed=RANDOM_SEED,
            output_with_seed=False,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
