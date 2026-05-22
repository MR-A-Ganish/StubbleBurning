import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1. LOAD ML-PREDICTED HISTORICAL DATA
# --------------------------------------------------
INPUT_FILE = "india_stubble_ml_predictions.csv"
OUTPUT_FORECAST = "india_stubble_forecast_2025.csv"

print("📥 Loading ML prediction data...")
df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# 2. AGGREGATE SEASONAL TRENDS PER GRID
# --------------------------------------------------
trend_df = (
    df.groupby("grid_id")
    .agg(
        mean_fires=("fires_per_day", "mean"),
        mean_brightness=("avg_brightness_scaled", "mean"),
        mean_frp=("avg_frp_scaled", "mean"),
        trend_fire=("fires_per_day", lambda x: x.iloc[-1] - x.iloc[0]),
        lat=("lat_center", "mean"),
        lon=("lon_center", "mean")
    )
    .reset_index()
)

# --------------------------------------------------
# 3. PREPARE FEATURES FOR FORECAST
# --------------------------------------------------
features = [
    "mean_fires",
    "mean_brightness",
    "mean_frp",
    "trend_fire"
]

X = trend_df[features].fillna(0)

# --------------------------------------------------
# 4. TRAIN FORECAST MODEL
# --------------------------------------------------
# Target: historically high-risk grids
threshold = df["fire_count"].quantile(0.75)
df["historical_high_risk"] = (df["fire_count"] >= threshold).astype(int)

train_df = df.groupby("grid_id")["historical_high_risk"].max().reset_index()
trend_df = trend_df.merge(train_df, on="grid_id", how="left")

y = trend_df["historical_high_risk"].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=500)
model.fit(X_scaled, y)

# --------------------------------------------------
# 5. FORECAST NEXT SEASON RISK
# --------------------------------------------------
trend_df["forecast_risk_probability"] = model.predict_proba(X_scaled)[:, 1]

trend_df["forecast_risk_label"] = trend_df[
    "forecast_risk_probability"
].apply(lambda x: "High Risk" if x >= 0.6 else "Low Risk")

# --------------------------------------------------
# 6. SAVE FORECAST
# --------------------------------------------------
trend_df.to_csv(OUTPUT_FORECAST, index=False)

print("🔮 Forecast saved:", OUTPUT_FORECAST)
print("High-risk grids forecasted:",
      (trend_df["forecast_risk_label"] == "High Risk").sum())
