# Derive soil-depth differences, averages, and cyclical time features from sensor readings.
# The same transformations are used for training data and live predictions.

import numpy as np
import pandas as pd

from config import SPLIT_DIR, PROCESSED_DIR, FEATURE_DIR, MODEL_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from the raw sensor readings, including soil-depth differences, averages, and cyclical time features.
    The same transformations are applied to both training data and live predictions.
    """


    # Work on a copy so callers keep their original sensor table unchanged.
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # L1 and L2 are the two soil depths; positive differences mean L1 is higher than L2.
    df["moisture_depth_difference"] = df["soil_moisture_l1"] - df["soil_moisture_l2"]
    df["temperature_depth_difference"] = df["soil_temp_l1_c"] - df["soil_temp_l2_c"]
    # Average the two soil depths to reduce noise and missing values; this is a common practice in soil science.
    df["avg_soil_moisture"] = df[["soil_moisture_l1", "soil_moisture_l2"]].mean(axis=1)
    df["avg_soil_temp_c"] = df[["soil_temp_l1_c", "soil_temp_l2_c"]].mean(axis=1)
    # Air-to-soil temperature relationship.
    df["air_soil_temp_difference"] = df["air_temp_c"] - df["avg_soil_temp_c"]
    # Convert hours into cyclical values (0–23 hours repeat daily).
    # This helps the model understand that 23:00 and 00:00 are close.
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ["train", "validation", "test"]:
        path = SPLIT_DIR / f"{split}.csv"
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = add_features(df)
        frames.append(df)
        output_columns = ["timestamp"] + MODEL_FEATURES
        df[output_columns].to_csv(FEATURE_DIR / f"{split}_engineered.csv", index=False)
        print(f"{split}: {len(df):,} rows, " f"{len(MODEL_FEATURES)} model features")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Also save a combined table for inspection; training still uses the separate chronological splits.
    pd.concat(frames, ignore_index=True).to_csv(PROCESSED_DIR / "engineered_dataset.csv", index=False)
    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()
