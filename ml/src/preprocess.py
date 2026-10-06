from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    SPLIT_DIR,
    PREPROCESS_RESULTS_DIR,
    SHEET_NAME,
    RAW_COLUMNS,
    COLUMN_MAP,
    BASE_FEATURES,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


def find_workbook() -> Path:
    preferred = [
        "hrzsi_dataset_and_analysis-1-1-object_df_raw.xlsx",
        "HRZSI_Dataset_and_Analysis (1).xlsx",
        "HRZSI_Dataset_and_Analysis.xlsx",
    ]

    for name in preferred:
        path = RAW_DIR / name
        if path.exists():
            return path

    files = sorted(RAW_DIR.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(
            f"No .xlsx file found in {RAW_DIR}"
        )

    return files[0]


def load_raw_data(path: Path) -> pd.DataFrame:
    excel = pd.ExcelFile(path)

    if SHEET_NAME in excel.sheet_names:
        sheet = SHEET_NAME
    else:
        # This supports the workbook exported with only Object_df_raw.
        sheet = excel.sheet_names[0]

    df = pd.read_excel(path, sheet_name=sheet)

    missing = [
        column for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    return df[RAW_COLUMNS].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)

    # Parse timestamp.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # The Excel file contains values such as 12:59:59.999.
    # Round them to the intended hourly timestamp.
    df["timestamp"] = df["timestamp"].dt.round("h")

    # Convert sensor columns to numeric.
    for column in BASE_FEATURES:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Remove invalid timestamp rows.
    df = df.dropna(subset=["timestamp"])

    # Sort and remove duplicate timestamps after rounding.
    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # Broad physical validity checks.
    valid_ranges = {
        "air_temp_c": (0, 50),
        "humidity": (0, 100),
        "soil_moisture_l1": (0, 100),
        "soil_temp_l1_c": (0, 50),
        "soil_moisture_l2": (0, 100),
        "soil_temp_l2_c": (0, 50),
    }

    for column, (minimum, maximum) in valid_ranges.items():
        invalid = (
            df[column].notna()
            & ~df[column].between(minimum, maximum)
        )
        df.loc[invalid, column] = np.nan

    # The EDA showed no missing values in the original six sensors.
    # This remains defensive in case the source workbook changes.
    df = df.dropna(
        subset=BASE_FEATURES
    ).reset_index(drop=True)

    return df


def chronological_split(df: pd.DataFrame):
    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))
    if not 0 < train_end < validation_end < n:
        raise ValueError("Dataset is too small for three nonempty chronological splits")
    return {
        "train": df.iloc[:train_end].copy(),
        "validation": df.iloc[train_end:validation_end].copy(),
        "test": df.iloc[validation_end:].copy(),
    }


def main():
    raw = load_raw_data(find_workbook())
    cleaned = clean_data(raw)
    splits = chronological_split(cleaned)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(PROCESSED_DIR / "cleaned_dataset.csv", index=False)
    for name, frame in splits.items():
        frame.to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(frame):,} rows")
    pd.DataFrame([{
        "raw_rows": len(raw), "cleaned_rows": len(cleaned),
        "removed_rows": len(raw) - len(cleaned),
        **{f"{name}_rows": len(frame) for name, frame in splits.items()},
    }]).to_csv(PREPROCESS_RESULTS_DIR / "summary.csv", index=False)


if __name__ == "__main__":
    main()
