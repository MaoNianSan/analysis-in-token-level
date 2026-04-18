import os
import io
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
    make_scorer,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from config import (
    RANDOM_SEED,
    TRAIN_CLASSIFIER_DIRS,
    TRAIN_AD_PATH,
    TRAIN_CONTROL_PATH,
    TEST_PATH,
    VECTORIZER_PATH,
    MODEL_PATH,
    CLASSIFICATION_RESULTS_PATH,
    TRAIN_CLASSIFIER_OUTPUT_PATH,
    ensure_dirs,
    set_global_seed,
)

# ----------------------------
# Fixed evaluation settings
# ----------------------------
CV_N_SPLITS = 5
CV_N_REPEATS = 3
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

FINAL_CONFIG = {
    "tfidf": {
        "analyzer": "word",
        "ngram_range": [1, 2],
        "max_features": None,
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
        "use_idf": False,
        "norm": None,
        "stop_words": None,
        "lowercase": True,
        "token_pattern": r"(?u)\b\w+\b",
    },
    "lr": {
        "C": 0.1,
        "penalty": "l2",
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 5000,
        "random_state": 42,
    },
    "threshold": 0.47999999999999987,
}


def add_seed_suffix(path, seed):
    """Append seed suffix before file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}_seed{seed}{ext}"


def build_vectorizer():
    tfidf_config = FINAL_CONFIG["tfidf"]
    return TfidfVectorizer(
        analyzer=tfidf_config["analyzer"],
        ngram_range=tuple(tfidf_config["ngram_range"]),
        max_features=tfidf_config["max_features"],
        min_df=tfidf_config["min_df"],
        max_df=tfidf_config["max_df"],
        sublinear_tf=tfidf_config["sublinear_tf"],
        use_idf=tfidf_config["use_idf"],
        norm=tfidf_config["norm"],
        stop_words=tfidf_config["stop_words"],
        lowercase=tfidf_config["lowercase"],
        token_pattern=tfidf_config["token_pattern"],
    )


def build_classifier():
    lr_config = FINAL_CONFIG["lr"]

    # Handle deprecated penalty parameter for newer scikit-learn versions
    if lr_config["penalty"] == "l2":
        # For L2 penalty, don't specify penalty parameter (use default)
        return LogisticRegression(
            C=lr_config["C"],
            solver=lr_config["solver"],
            class_weight=lr_config["class_weight"],
            max_iter=lr_config["max_iter"],
            random_state=lr_config["random_state"],
        )
    else:
        # For other penalties, specify it explicitly
        return LogisticRegression(
            C=lr_config["C"],
            penalty=lr_config["penalty"],
            solver=lr_config["solver"],
            class_weight=lr_config["class_weight"],
            max_iter=lr_config["max_iter"],
            random_state=lr_config["random_state"],
        )


def compute_stratified_bootstrap_ci(
    metric_func,
    y_true,
    y_pred_or_prob,
    n_resamples=BOOTSTRAP_RESAMPLES,
    confidence_level=CONFIDENCE_LEVEL,
    random_state=RANDOM_SEED,
):
    """
    Stratified bootstrap CI for binary classification metrics.
    Safer than naive bootstrap for small samples / AUC.
    """
    y_true = np.asarray(y_true)
    y_pred_or_prob = np.asarray(y_pred_or_prob)

    classes = np.unique(y_true)
    rng = np.random.default_rng(random_state)
    class_indices = {c: np.where(y_true == c)[0] for c in classes}

    stats = []
    for _ in range(n_resamples):
        sampled_indices = []
        for c in classes:
            idx = class_indices[c]
            sampled_c = rng.choice(idx, size=len(idx), replace=True)
            sampled_indices.append(sampled_c)

        sampled_indices = np.concatenate(sampled_indices)
        rng.shuffle(sampled_indices)

        try:
            value = metric_func(
                y_true[sampled_indices], y_pred_or_prob[sampled_indices]
            )
            if np.isfinite(value):
                stats.append(value)
        except ValueError:
            continue

    if len(stats) == 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence_level
    low = np.percentile(stats, 100 * (alpha / 2.0))
    high = np.percentile(stats, 100 * (1.0 - alpha / 2.0))
    return low, high


def evaluate_predictions(y_true, y_pred, y_prob):
    """
    Evaluate binary classification metrics.
    Assumes Control=0, AD=1.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    specificity = 0.0
    if (cm[0, 0] + cm[0, 1]) > 0:
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),  # sensitivity
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": brier_score_loss(y_true, y_prob),
    }
    return metrics, cm, cm_normalized


def evaluate_cross_validation_on_texts(
    X_text,
    y,
    n_splits=CV_N_SPLITS,
    n_repeats=CV_N_REPEATS,
    random_state=RANDOM_SEED,
):
    """
    Repeated stratified CV on raw texts using Pipeline(TFIDF + LR).
    This avoids vectorizer leakage.
    """
    pipeline = Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("clf", build_classifier()),
        ]
    )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "mcc": make_scorer(matthews_corrcoef),
        "neg_log_loss": "neg_log_loss",
        "neg_brier_score": "neg_brier_score",
    }

    cv_scores = cross_validate(
        pipeline,
        X_text,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
        return_train_score=False,
    )

    results = {}
    for key, values in cv_scores.items():
        if not key.startswith("test_"):
            continue

        metric_name = key.replace("test_", "")
        values = np.asarray(values)

        if metric_name == "neg_log_loss":
            results["log_loss"] = {
                "mean": float((-values).mean()),
                "std": float(values.std()),
            }
        elif metric_name == "neg_brier_score":
            results["brier"] = {
                "mean": float((-values).mean()),
                "std": float(values.std()),
            }
        else:
            results[metric_name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

    return results


def print_metrics_block(metrics, accuracy_ci=None, auc_ci=None):
    print("Evaluation Metrics")
    print("-" * 60)

    if accuracy_ci is not None and not np.isnan(accuracy_ci[0]):
        print(
            f"Accuracy:              {metrics['accuracy']:.4f} "
            f"(95% CI: {accuracy_ci[0]:.4f} - {accuracy_ci[1]:.4f})"
        )
    else:
        print(f"Accuracy:              {metrics['accuracy']:.4f}")

    print(f"Balanced Accuracy:     {metrics['balanced_accuracy']:.4f}")
    print(f"Precision (AD):        {metrics['precision']:.4f}")
    print(f"Recall/Sensitivity:    {metrics['recall']:.4f}")
    print(f"Specificity:           {metrics['specificity']:.4f}")
    print(f"F1-Score (AD):         {metrics['f1']:.4f}")

    if auc_ci is not None and not np.isnan(auc_ci[0]):
        print(
            f"ROC-AUC:               {metrics['roc_auc']:.4f} "
            f"(95% CI: {auc_ci[0]:.4f} - {auc_ci[1]:.4f})"
        )
    else:
        print(f"ROC-AUC:               {metrics['roc_auc']:.4f}")

    print(f"PR-AUC:                {metrics['average_precision']:.4f}")
    print(f"MCC:                   {metrics['mcc']:.4f}")
    print(f"Log Loss:              {metrics['log_loss']:.4f}")
    print(f"Brier Score:           {metrics['brier']:.4f}")


def print_confusion_and_report(y_true, y_pred, cm, cm_normalized):
    print("\n" + "-" * 60)
    print("Confusion Matrix (Raw)")
    print("-" * 60)
    print(cm)

    print("\n" + "-" * 60)
    print("Confusion Matrix (Row-normalized)")
    print("-" * 60)
    print(cm_normalized)

    print("\n" + "-" * 60)
    print("Classification Report")
    print("-" * 60)
    print(classification_report(y_true, y_pred, target_names=["Control", "AD"]))


def save_prediction_csv(
    df, y_true, y_pred, y_prob, y_decision, output_path, split_name
):
    result_cols = [col for col in ["sample_id", "mmse", "Speech"] if col in df.columns]
    results_df = df[result_cols].copy()
    results_df["true_label"] = y_true
    results_df["predicted_label"] = y_pred
    results_df["ad_score"] = y_prob
    results_df["decision_score"] = y_decision
    results_df["correct"] = y_true == y_pred
    results_df["dataset_split"] = split_name
    results_df.to_csv(output_path, index=False)


def main(seed=RANDOM_SEED, output_with_seed=False):
    set_global_seed(seed)
    ensure_dirs(TRAIN_CLASSIFIER_DIRS, keys=["model", "predictions", "logs"])

    vectorizer_path = (
        add_seed_suffix(VECTORIZER_PATH, seed) if output_with_seed else VECTORIZER_PATH
    )
    model_path = add_seed_suffix(MODEL_PATH, seed) if output_with_seed else MODEL_PATH
    classification_results_path = (
        add_seed_suffix(CLASSIFICATION_RESULTS_PATH, seed)
        if output_with_seed
        else CLASSIFICATION_RESULTS_PATH
    )
    train_classifier_output_path = (
        add_seed_suffix(TRAIN_CLASSIFIER_OUTPUT_PATH, seed)
        if output_with_seed
        else TRAIN_CLASSIFIER_OUTPUT_PATH
    )

    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    # --------------------------------------------------
    # New data logic:
    # 1) test.csv = development / training set
    # 2) ad.csv + control.csv = external evaluation set
    # --------------------------------------------------
    dev_df = pd.read_csv(TEST_PATH)
    ad_df = pd.read_csv(TRAIN_AD_PATH)
    control_df = pd.read_csv(TRAIN_CONTROL_PATH)
    external_df = pd.concat([ad_df, control_df], ignore_index=True)

    dev_df["Speech"] = dev_df["Speech"].fillna("")
    external_df["Speech"] = external_df["Speech"].fillna("")

    X_dev_text = dev_df["Speech"]
    y_dev = dev_df["label"].values

    X_external_text = external_df["Speech"]
    y_external = external_df["label"].values

    # ----------------------------
    # Internal validation on test.csv
    # ----------------------------
    cv_results = evaluate_cross_validation_on_texts(
        X_text=X_dev_text,
        y=y_dev,
        n_splits=CV_N_SPLITS,
        n_repeats=CV_N_REPEATS,
        random_state=RANDOM_SEED,
    )

    # ----------------------------
    # Fit final model on full development set
    # ----------------------------
    vectorizer = build_vectorizer()
    clf = build_classifier()

    X_dev = vectorizer.fit_transform(X_dev_text)
    clf.fit(X_dev, y_dev)

    # Development-set resubstitution (debug only)
    y_dev_pred = clf.predict(X_dev)
    y_dev_prob = clf.predict_proba(X_dev)[:, 1]
    y_dev_decision = clf.decision_function(X_dev)

    dev_metrics, dev_cm, dev_cm_normalized = evaluate_predictions(
        y_dev, y_dev_pred, y_dev_prob
    )
    dev_auc_ci = compute_stratified_bootstrap_ci(roc_auc_score, y_dev, y_dev_prob)
    dev_accuracy_ci = compute_stratified_bootstrap_ci(accuracy_score, y_dev, y_dev_pred)

    # ----------------------------
    # External evaluation on ad.csv + control.csv
    # ----------------------------
    threshold = FINAL_CONFIG["threshold"]

    X_external = vectorizer.transform(X_external_text)
    y_external_prob = clf.predict_proba(X_external)[:, 1]
    y_external_pred = (y_external_prob >= threshold).astype(int)  # Use tuned threshold
    y_external_decision = clf.decision_function(X_external)

    external_metrics, external_cm, external_cm_normalized = evaluate_predictions(
        y_external, y_external_pred, y_external_prob
    )
    external_auc_ci = compute_stratified_bootstrap_ci(
        roc_auc_score, y_external, y_external_prob
    )
    external_accuracy_ci = compute_stratified_bootstrap_ci(
        accuracy_score, y_external, y_external_pred
    )

    # ----------------------------
    # Print results
    # ----------------------------
    print("=" * 60)
    print("DEVELOPMENT SECTION (test.csv)")
    print("=" * 60)
    print(f"Development set size: {len(dev_df)}")
    print("Model: TF-IDF + Logistic Regression")
    print(f"Number of features: {X_dev.shape[1]}")

    print("\n" + "-" * 60)
    print(
        f"Internal Validation Results (Repeated Stratified "
        f"{CV_N_SPLITS}-fold CV, {CV_N_REPEATS} repeats)"
    )
    print("-" * 60)
    ordered_cv_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "average_precision",
        "mcc",
        "log_loss",
        "brier",
    ]
    for metric_name in ordered_cv_metrics:
        if metric_name in cv_results:
            values = cv_results[metric_name]
            print(
                f"{metric_name.capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}"
            )

    print("\n" + "-" * 60)
    print("Development-Set Resubstitution (debug only; optimistic)")
    print("-" * 60)
    print_metrics_block(dev_metrics, accuracy_ci=dev_accuracy_ci, auc_ci=dev_auc_ci)
    print_confusion_and_report(y_dev, y_dev_pred, dev_cm, dev_cm_normalized)

    print("\n" + "=" * 60)
    print("EXTERNAL TEST SECTION (ad.csv + control.csv)")
    print("=" * 60)
    print(f"Number of samples: {len(external_df)}")
    print(f"Decision threshold: {threshold:.4f}")
    print("\n" + "-" * 60)
    print_metrics_block(
        external_metrics,
        accuracy_ci=external_accuracy_ci,
        auc_ci=external_auc_ci,
    )
    print_confusion_and_report(
        y_external,
        y_external_pred,
        external_cm,
        external_cm_normalized,
    )

    print("\n" + "-" * 60)
    print("Sample-wise Results (External Test):")
    for idx, row in external_df.iterrows():
        sample_id = row.get("sample_id", idx)
        true_label = row["label"]
        pred_label = y_external_pred[idx]
        prob = y_external_prob[idx]
        print(
            f"Sample {sample_id}: True={true_label}, Pred={pred_label}, Prob={prob:.4f}"
        )

    # Restore stdout and save outputs
    sys.stdout = old_stdout
    output_content = captured_output.getvalue()
    # Don't print output_content here as it was already captured and will be printed above

    with open(train_classifier_output_path, "w", encoding="utf-8") as f:
        f.write(output_content)

    # Save model artifacts
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)

    # Keep original output path for external test results
    save_prediction_csv(
        df=external_df,
        y_true=y_external,
        y_pred=y_external_pred,
        y_prob=y_external_prob,
        y_decision=y_external_decision,
        output_path=classification_results_path,
        split_name="external_test",
    )

    # Extra debug file for development-set predictions
    base, ext = os.path.splitext(classification_results_path)
    dev_debug_results_path = f"{base}_development_debug{ext}"
    save_prediction_csv(
        df=dev_df,
        y_true=y_dev,
        y_pred=y_dev_pred,
        y_prob=y_dev_prob,
        y_decision=y_dev_decision,
        output_path=dev_debug_results_path,
        split_name="development_resubstitution",
    )

    print("\nSaved files:")
    print(f"- {vectorizer_path}")
    print(f"- {model_path}")
    print(f"- {classification_results_path} (external test results)")
    print(f"- {dev_debug_results_path} (development debug results)")
    print(f"- {train_classifier_output_path}")


if __name__ == "__main__":
    main()
