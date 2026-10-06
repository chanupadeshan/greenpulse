# GreenPulse AI — ML

Hourly greenhouse sensor anomaly detection using Isolation Forest, a dense autoencoder,
and an LSTM autoencoder. Neural models use PyTorch `.pt` weights; scalers and the
Isolation Forest use joblib. Run commands from `ml/`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
python src/feature_engineering.py
python src/hyperparameter.py --quick
python src/train_isolation_forest.py
python src/train_autoencoder.py
python src/train_lstm_autoencoder.py
python src/evaluate_models.py
python src/select_best_model.py
python src/predict_anomaly.py data/splits/test.csv
python -m unittest discover -s tests -v
```

Remove `--quick` for the wider tuning search. Neural training accepts `--epochs N`
for smoke runs. Default training uses up to 80 dense / 70 LSTM epochs with early
stopping. CPU thread limits such as `OMP_NUM_THREADS=2` can help on shared machines.

Preprocessing rounds timestamps to hours, removes duplicate timestamps and readings
outside the configured physical ranges, then splits chronologically 70/15/15.
Feature engineering writes `*_engineered.csv` beside the base splits. Scalers fit
only the training split. Tuning uses validation data; evaluation uses test data.
LSTM sequences exclude gaps and require consecutive hourly readings.

Training generates `models/<name>/metadata.json`, `scaler.joblib`, and `model.joblib`
(Isolation Forest) or `model.pt` (autoencoders). Tuning writes
`models/tuning/<name>_best_params.json`. Evaluation writes
`results/evaluation/model_comparison.csv`; selection writes
`models/selected_model.json`. Missing trained artifacts require training first.

Live input requires `timestamp`, `air_temp_c`, `humidity`, `soil_moisture_l1`,
`soil_temp_l1_c`, `soil_moisture_l2`, and `soil_temp_l2_c`. Prediction scores the
latest row, or the latest full window for LSTM. Override selection with
`--model isolation_forest`, `--model autoencoder`, or `--model lstm_autoencoder`.

No ground-truth anomaly labels are available. Reported classification metrics measure
injected synthetic anomalies against observed data assumed normal, and do not establish
real-world detection accuracy. Thresholds use the 99th percentile of validation scores.
