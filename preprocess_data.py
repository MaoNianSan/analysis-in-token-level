import os
import re
import pandas as pd

from config import DATA_DIR, PREPROCESS_DATA_DIRS, ensure_dirs


RAW_TO_PROCESSED = {
    "ad_s2t_wav2vec.csv": "ad.csv",
    "control_s2t_wav2vec.csv": "control.csv",
    "test_s2t_wav2vec.csv": "test.csv",
}


def extract_sample_id(file_path):
    """
    Preserve the raw sample identifier if it can be extracted from the filename.
    No zero-padding or forced 3-digit normalization.
    """
    if pd.isna(file_path):
        return None

    filename = os.path.basename(str(file_path))
    match = re.search(r"(adrso\d+|adrsdt\d+)", filename, flags=re.IGNORECASE)
    if not match:
        return None

    token = match.group(1)
    numeric = re.search(r"(\d+)", token)
    if numeric:
        value = numeric.group(1)
        try:
            return int(value)
        except ValueError:
            return value
    return token


def ensure_sample_id(df):
    """Keep existing sample_id when available; otherwise fill from file path."""
    if "sample_id" not in df.columns:
        df["sample_id"] = pd.NA

    if "file" in df.columns:
        extracted = df["file"].apply(extract_sample_id)
        df["sample_id"] = df["sample_id"].where(df["sample_id"].notna(), extracted)

    return df


def main():
    ensure_dirs(PREPROCESS_DATA_DIRS, keys=["csv"])
    processed_frames = []

    for raw_name, output_name in RAW_TO_PROCESSED.items():
        raw_path = os.path.join(DATA_DIR, raw_name)
        if not os.path.exists(raw_path):
            print(f"File {raw_path} not found")
            continue

        df = pd.read_csv(raw_path)
        df = ensure_sample_id(df)

        if "file" in df.columns:
            df = df.drop(columns=["file"])

        output_path = os.path.join(PREPROCESS_DATA_DIRS["csv"], output_name)
        df.to_csv(output_path, index=False)
        processed_frames.append(df)

        print(f"Processed {raw_name} -> {output_path}")

    if processed_frames:
        combined_df = pd.concat(processed_frames, ignore_index=True)
        combined_path = os.path.join(
            PREPROCESS_DATA_DIRS["csv"], "combined_processed_data.csv"
        )
        combined_df.to_csv(combined_path, index=False)
        print(f"Combined processed data -> {combined_path}")


if __name__ == "__main__":
    main()
