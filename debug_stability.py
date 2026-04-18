import os
import warnings
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    RANDOM_SEED,
    ROBUSTNESS_SEEDS,
    WINDOW_LENGTH,
    STRIDE,
    TOP_K,
    NOISE_SCALE,
    VECTORIZER_PATH,
    MODEL_PATH,
    DATASET_PATHS,
    build_dataset_dirs,
    ensure_dirs,
    EPSILON,
    set_global_seed,
    compute_direction,
)

warnings.filterwarnings("ignore")


def add_seed_suffix(path, seed):
    base, ext = os.path.splitext(path)
    return f"{base}_seed{seed}{ext}"


def resolve_dataset_path(dataset_name, dataset_path):
    if dataset_path is not None:
        return dataset_path
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_PATHS[dataset_name]


def get_score(text, vectorizer, clf, rng=None, noise_scale=0.0):
    if not str(text).strip():
        return 0.5
    X_dense = vectorizer.transform([str(text)]).toarray()
    if rng is not None and noise_scale > 0:
        X_dense = X_dense + rng.normal(0, noise_scale, X_dense.shape)
    return float(clf.predict_proba(X_dense)[0, 1])


def get_windows(tokens):
    windows = []
    for start in range(0, len(tokens) - WINDOW_LENGTH + 1, STRIDE):
        windows.append((start, start + WINDOW_LENGTH))
    return windows


def analyze_sample(text, vectorizer, clf, seed):
    tokens = str(text).split()
    if len(tokens) < WINDOW_LENGTH:
        return []

    rng = np.random.default_rng(seed)
    original_prob = get_score(text, vectorizer, clf, rng, NOISE_SCALE)

    results = []

    for start, end in get_windows(tokens):
        text_wo = " ".join(tokens[:start] + tokens[end:])
        score_wo = get_score(text_wo, vectorizer, clf, rng, NOISE_SCALE)

        delta = original_prob - score_wo

        results.append(
            {
                "window": (start, end),
                "delta": delta,
                "abs_delta": abs(delta),
                "direction": compute_direction(delta, EPSILON),
            }
        )

    results = sorted(results, key=lambda x: x["abs_delta"], reverse=True)[:TOP_K]
    return results


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main(
    seed=RANDOM_SEED,
    output_with_seed=False,
    dataset_name="test",
    dataset_path=None,
):
    set_global_seed(seed)

    dataset_path = resolve_dataset_path(dataset_name, dataset_path)
    dirs = build_dataset_dirs("debug_stability", dataset_name)
    ensure_dirs(dirs, keys=["csv"])

    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)

    df = pd.read_csv(dataset_path)
    df["Speech"] = df["Speech"].fillna("")

    print("=" * 80)
    print(f"Stability | dataset={dataset_name}")
    print("=" * 80)

    rows = []

    for idx, row in df.iterrows():
        text = str(row["Speech"]).strip()
        if not text:
            continue

        ref = analyze_sample(text, vectorizer, clf, RANDOM_SEED)
        ref_windows = [w["window"] for w in ref]

        for compare_seed in ROBUSTNESS_SEEDS:
            cur = analyze_sample(text, vectorizer, clf, compare_seed)
            cur_windows = [w["window"] for w in cur]

            overlap = jaccard(ref_windows, cur_windows)

            rows.append(
                {
                    "dataset": dataset_name,
                    "sample_id": row.get("sample_id", idx),
                    "reference_seed": RANDOM_SEED,
                    "compare_seed": compare_seed,
                    "topk_overlap": overlap,
                }
            )

    result_df = pd.DataFrame(rows)

    out_path = Path(dirs["csv"]) / "stability_summary.csv"

    if output_with_seed:
        out_path = Path(add_seed_suffix(str(out_path), seed))

    result_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")

    return {
        "dataset": dataset_name,
        "seed": seed,
        "n_rows": len(result_df),
    }


if __name__ == "__main__":
    for dataset_name, dataset_path in DATASET_PATHS.items():
        main(
            seed=RANDOM_SEED,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
