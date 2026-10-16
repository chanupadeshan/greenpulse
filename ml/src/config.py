# Shared paths, sensor columns, model features, and dataset split settings.
# Used by both offline training scripts and live anomaly prediction.

from pathlib import Path

# Build paths relative to this file, so scripts can locate ml/ regardless of the working directory.
ML_DIR = Path(__file__).resolve().parent.parent

# Keep original data, cleaned data, split datasets, and trained model files in separate folders.
RAW_DIR = ML_DIR / "data" / "raw"
PROCESSED_DIR = ML_DIR / "data" / "processed"
SPLIT_DIR = ML_DIR / "data" / "splits"
FEATURE_DIR = SPLIT_DIR
MODEL_DIR = ML_DIR / "models"

PREPROCESS_RESULTS_DIR = ML_DIR / "results" / "preprocessing"
TUNING_RESULTS_DIR = ML_DIR / "results" / "tuning"
EVALUATION_RESULTS_DIR = ML_DIR / "results" / "evaluation"

# Name of the Excel worksheet containing the original sensor observations.
SHEET_NAME = "Object_df_raw"

RAW_COLUMNS = ["DATE", "Air temperature (°C)", "Relative humidity (%)", "SMC1", "Ts1", "SMC2", "Ts2"]

# Translate spreadsheet headings into the consistent names used throughout the Python code.
COLUMN_MAP = {
    "DATE": "timestamp",
    "Air temperature (°C)": "air_temp_c",
    "Relative humidity (%)": "humidity",
    "SMC1": "soil_moisture_l1",
    "Ts1": "soil_temp_l1_c",
    "SMC2": "soil_moisture_l2",
    "Ts2": "soil_temp_l2_c",
}

# The six measurements supplied by the greenhouse: air temperature, humidity, and two soil depths.
# Temperatures use degrees Celsius; humidity and soil moisture use percentages.
BASE_FEATURES = [
    "air_temp_c",
    "humidity",
    "soil_moisture_l1",
    "soil_temp_l1_c",
    "soil_moisture_l2",
    "soil_temp_l2_c",
]

# The model receives these columns in exactly this order: six readings plus seven calculated features.
# Changing this list requires retraining, because saved scalers and weights expect the same inputs.
MODEL_FEATURES = BASE_FEATURES + [
    "moisture_depth_difference",
    "temperature_depth_difference",
    "avg_soil_moisture",
    "avg_soil_temp_c",
    "air_soil_temp_difference",
    "hour_sin",
    "hour_cos",
]

# Use the oldest 70% to learn, the next 15% to tune settings, and the newest 15% to evaluate.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Reuse a fixed random seed to make experiments easier to reproduce.
RANDOM_STATE = 42
