# Alzheimer's Disease Speech Analysis Pipeline

A reproducible TF-IDF-based pipeline for Alzheimer's Disease (AD) speech-text analysis.

The repository covers preprocessing, subject-level classification, sliding-window importance analysis, token attribution, stability diagnostics, and figure generation. Outputs are stored by script and by dataset, so different stages do not overwrite each other.

## Repository Structure

- `main_pipeline.py` — runs the end-to-end pipeline
- `config.py` — shared configuration, path builders, and seed settings
- `preprocess_data.py` — preprocesses raw CSV files and extracts `sample_id`
- `train_classifier.py` — trains TF-IDF + Logistic Regression and reports evaluation
- `window_extraction.py` — computes sliding-window deletion importance
- `token_aggregation.py` — computes token attribution within selected windows and aggregates token statistics
- `attribution_methods.py` — token attribution helpers
- `debug_stability.py` — compares top-window overlap across seeds
- `visualization.py` — generates plots from stage outputs
- `data_Test AD/` — raw data plus generated outputs
- `requirements.txt` — dependencies

## Input Data

Put the raw files under `data_Test AD/`:

- `ad_s2t_wav2vec.csv`
- `control_s2t_wav2vec.csv`
- `test_s2t_wav2vec.csv`

## Output Layout

Each script writes into its own folder under `data_Test AD/`.

- `data_Test AD/preprocess_data/`
- `data_Test AD/train_classifier/`
- `data_Test AD/window_extraction/<dataset>/`
- `data_Test AD/token_aggregation/<dataset>/`
- `data_Test AD/debug_stability/<dataset>/`
- `data_Test AD/visualization/<dataset>/`
- `data_Test AD/main_pipeline/`

Here `<dataset>` is one of `test`, `ad`, or `control`.

Directory creation is lazy:
- `config.py` only builds paths and does not create folders at import time.
- Folders are created by each script only for the subdirectories it actually writes.
- The pipeline no longer pre-creates `csv/figures/logs/model/metrics/predictions` for every script/dataset.

## What Each Stage Does

### 1. `preprocess_data.py`

Reads the three raw CSV files, keeps or reconstructs `sample_id`, removes the original `file` column, and saves:

- `data_Test AD/preprocess_data/csv/ad.csv`
- `data_Test AD/preprocess_data/csv/control.csv`
- `data_Test AD/preprocess_data/csv/test.csv`
- `data_Test AD/preprocess_data/csv/combined_processed_data.csv`

### 2. `train_classifier.py`

Trains a subject-level TF-IDF + Logistic Regression classifier.

Current code logic:

- `test.csv` is used as the development set
- `ad.csv` + `control.csv` are concatenated as the external evaluation set
- repeated stratified 5-fold CV with 3 repeats is reported on the development set
- the final classifier is fit on the full development set

Main outputs:

- `data_Test AD/train_classifier/model/tfidf_vectorizer.joblib`
- `data_Test AD/train_classifier/model/logistic_regression_model.joblib`
- `data_Test AD/train_classifier/predictions/classification_results.csv`
- `data_Test AD/train_classifier/logs/train_classifier_output.txt`

### 3. `window_extraction.py`

For each dataset, slides a fixed window over each transcript and deletes one window at a time.

Current settings from `config.py`:

- window length = 4 tokens
- stride = 2 tokens

For each window, the script records:

- `delta_window = original_prob - score_without_window`
- `abs_delta_window`
- `direction_window` (`AD_support`, `Control_support`, or `Neutral`)
- energy-distance style distribution diagnostics

Main outputs for each dataset:

- `data_Test AD/window_extraction/<dataset>/csv/window_importance.csv`
- `data_Test AD/window_extraction/<dataset>/logs/window_extraction_output.txt`

When `output_with_seed=True`, seed-specific files such as `window_importance_seed123.csv` are also created.

### 4. `token_aggregation.py`

Loads the window output for one dataset, selects candidate windows, runs token-level leave-one-out attribution, and saves:

- `data_Test AD/token_aggregation/<dataset>/csv/candidate_windows_selected.csv`
- `data_Test AD/token_aggregation/<dataset>/csv/token_attribution_all.csv`
- `data_Test AD/token_aggregation/<dataset>/csv/token_attribution_aggregated.csv`

The aggregated file summarizes token-level importance and direction statistics.

### 5. `debug_stability.py`

Measures top-K window overlap across seeds for each sample and saves:

- `data_Test AD/debug_stability/<dataset>/csv/stability_summary.csv`

### 6. `visualization.py`

Reads window, token, and stability outputs and writes figures to:

- `data_Test AD/visualization/<dataset>/figures/`

If seed-specific inputs are used, figures are written under:

- `data_Test AD/visualization/<dataset>/figures/seed_<seed>/`

## Run the Full Pipeline

Default run:

```bash
python main_pipeline.py
```

Run one seed only:

```bash
python main_pipeline.py --seed 123
```

Run several specific seeds:

```bash
python main_pipeline.py --seeds 7,42,123
```

Run only selected datasets:

```bash
python main_pipeline.py --seed 123 --datasets test,ad
```

## How to Set Seed During Runtime

There are two supported ways.

### Option 1: pass seed(s) to `main_pipeline.py`

This is the easiest way for the full pipeline.

```bash
python main_pipeline.py --seed 123
python main_pipeline.py --seeds 7,42,123
```

### Option 2: use environment variables

This works for running individual scripts too.

Windows PowerShell:

```powershell
$env:RANDOM_SEED=123
$env:ROBUSTNESS_SEEDS="123,456,789"
python window_extraction.py
```

Windows CMD:

```cmd
set RANDOM_SEED=123
set ROBUSTNESS_SEEDS=123,456,789
python window_extraction.py
```

macOS / Linux:

```bash
RANDOM_SEED=123 ROBUSTNESS_SEEDS=123,456,789 python window_extraction.py
```

If no runtime override is provided, the defaults in `config.py` are used.

## Dependencies

```bash
pip install -r requirements.txt
```
