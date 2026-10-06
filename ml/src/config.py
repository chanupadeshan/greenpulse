from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = ML_DIR / "data" / "raw"
PROCESSED_DIR = ML_DIR / "data" / "processed"
SPLIT_DIR = ML_DIR / "data" / "splits"
FEATURE_DIR = SPLIT_DIR
MODEL_DIR = ML_DIR / "models"

PREPROCESS_RESULTS_DIR = ML_DIR / "results" / "preprocessing"
TUNING_RESULTS_DIR = ML_DIR / "results" / "tuning"
EVALUATION_RESULTS_DIR = ML_DIR / "results" / "evaluation"

SHEET_NAME = "Object_df_raw"

RAW_COLUMNS = [
    "DATE",
    "Air temperature (°C)",
    "Relative humidity (%)",
    "SMC1",
    "Ts1",
    "SMC2",
    "Ts2",
]

COLUMN_MAP = {
    "DATE": "timestamp",
    "Air temperature (°C)": "air_temp_c",
    "Relative humidity (%)": "humidity",
    "SMC1": "soil_moisture_l1",
    "Ts1": "soil_temp_l1_c",
    "SMC2": "soil_moisture_l2",
    "Ts2": "soil_temp_l2_c",
}

BASE_FEATURES = [
    "air_temp_c",
    "humidity",
    "soil_moisture_l1",
    "soil_temp_l1_c",
    "soil_moisture_l2",
    "soil_temp_l2_c",
]

MODEL_FEATURES = BASE_FEATURES + [
    "moisture_depth_difference",
    "temperature_depth_difference",
    "avg_soil_moisture",
    "avg_soil_temp_c",
    "air_soil_temp_difference",
    "hour_sin",
    "hour_cos",
]

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
