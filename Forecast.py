# --- 1. EXECUTE PRODUCTION FORECAST ---
# Darts automatically extracts the required history from the trailing edges of the TS inputs
prediction = model.predict(
    n=15, 
    series=target_ts,
    past_covariates=past_cov_ts,
    future_covariates=future_cov_ts
)

# --- 2. PARSE THE QUANTILE CHANNELS ---
# The prediction object holds all 3 requested quantiles (0.1, 0.5, 0.9)
# Isolate  90th percentile to build safety grid buffer
p90_power_forecast = prediction["facility_power_kw_0.9"]

# Extract the maximum predicted power value within 15-minute look-ahead window
max_predicted_p90_kw = p90_power_forecast.max().values()[0][0]
current_grid_capacity_kw = 600.0 # Hypothetical constraint threshold of current setup

# --- 3. AUTOMATION ACTION LOOP ---
if max_predicted_p90_kw > current_grid_capacity_kw:
    power_deficit = max_predicted_p90_kw - current_grid_capacity_kw
    print(f"[ALARM] Imminent AI cluster power surge detected!")
    print(f"Targeting p90 Max Load: {max_predicted_p90_kw:.2f} kW")
    print(f"Action: Initiating pre-emptive battery discharge of {power_deficit:.2f} kW NOW.")
    # Trigger API Call to Microgrid Battery Controller here ->
else:
    print(f"[NORMAL] Grid capacity sufficient. Expected p90 peak: {max_predicted_p90_kw:.2f} kW")
