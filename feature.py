import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# --------------------------------------------------
# 1. LOAD CLEAN DATA
# --------------------------------------------------
INPUT_FILE = "india_stubble_clean.csv"
OUTPUT_SCALED = "india_stubble_scaled.csv"
SCALER_FILE = "scaler.pkl"

print("📥 Loading clean dataset...")
df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# 2. SELECT FEATURES TO SCALE
# --------------------------------------------------
# These are continuous numerical features
scale_features = [
    "brightness",
    "frp",
    "latitude",
    "longitude",
    "dayofyear"
]

X = df[scale_features]

# --------------------------------------------------
# 3. APPLY STANDARD SCALING
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=[f"{c}_scaled" for c in scale_features]
)

# --------------------------------------------------
# 4. COMBINE WITH ORIGINAL DATA
# --------------------------------------------------
final_df = pd.concat(
    [df.reset_index(drop=True), scaled_df],
    axis=1
)

# --------------------------------------------------
# 5. SAVE OUTPUTS
# --------------------------------------------------
final_df.to_csv(OUTPUT_SCALED, index=False)
joblib.dump(scaler, SCALER_FILE)

print("✅ Feature scaling complete")
print("📁 Saved scaled dataset:", OUTPUT_SCALED)
print("💾 Saved scaler:", SCALER_FILE)
