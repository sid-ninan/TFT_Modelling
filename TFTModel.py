import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

# --- 1. SIMULATE SAMPLE DATA ---
# Assume 1-minute interval granularity data
idx = pd.date_range(start="2026-01-01", periods=1000, freq="1min")
df = pd.DataFrame(index=idx)

df["facility_power_kw"] = np.random.uniform(200, 800, size=1000)      # Target
df["ingress_network_mbps"] = np.random.uniform(50, 500, size=1000)   # Past Covariate
df["gpu_temp_c"] = np.random.uniform(40, 85, size=1000)              # Past Covariate
df["scheduled_cron_job"] = np.random.choice([0, 1], size=1000)       # Future Covariate
df["pue_weather_forecast"] = np.random.uniform(15, 35, size=1000)    # Future Covariate

# --- 2. CONVERT TO DARTS TIMESERIES ---
target_ts = TimeSeries.from_dataframe(df, value_cols="facility_power_kw")

# Past Covariates: Telemetry we only know up to the present moment
past_cov_ts = TimeSeries.from_dataframe(df, value_cols=["ingress_network_mbps", "gpu_temp_c"])

# Future Covariates: Information we know 15+ minutes in advance
future_cov_ts = TimeSeries.from_dataframe(df, value_cols=["scheduled_cron_job", "pue_weather_forecast"])

# --- 3. INITIALIZE THE TEMPORAL FUSION TRANSFORMER ---
model = TFTModel(
    input_chunk_length=60,           # Look back at the last 60 minutes of history
    output_chunk_length=15,          # Predict the next 15 minutes of power requirements
    hidden_size=32,                  # Capacity of the attention layers
    lstm_layers=1,                   # Processing sequential history
    num_attention_heads=4,           # Parallel attention mechanisms
    dropout=0.1,
    batch_size=32,
    n_epochs=15,
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]), # Triggers probabilistic outputs
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu"} # Tells PyTorch to run on local GPU infrastructure
)

# --- 4. TRAIN THE MODEL ---
# For a real implementation, you would split these into train/validation sets
model.fit(
    series=target_ts,
    past_covariates=past_cov_ts,
    future_covariates=future_cov_ts
)
