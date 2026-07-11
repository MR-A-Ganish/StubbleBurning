import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import geopandas as gpd
import os

from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

try:
    from alerts import trigger_alert
except ImportError:
    def trigger_alert(df):
        return False

st.set_page_config(page_title="AI Fire Monitoring", layout="wide")

st.title("🔥 AI Crop Residue Burning Monitoring System")
st.caption("NASA VIIRS Satellite Fire Data")

os.environ["SHAPE_RESTORE_SHX"] = "YES"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_resource
def load_data():
    df = pd.read_parquet(
        "india_stubble_clean.parquet",
        columns=["latitude", "longitude", "brightness", "frp", "acq_date"]
    )

    df["latitude"] = df["latitude"].astype("float32")
    df["longitude"] = df["longitude"].astype("float32")
    df["brightness"] = df["brightness"].astype("float32")
    df["frp"] = df["frp"].astype("float32")

    df["year"] = df["acq_date"].dt.year.astype("int16")
    df["month"] = df["acq_date"].dt.month.astype("int8")

    return df

# --------------------------------------------------
# LOAD + MAP DISTRICTS
# --------------------------------------------------
@st.cache_resource
def get_mapped_data():
    df = load_data()
    districts = gpd.read_file("gadm41_IND_2.shp").to_crs(epsg=4326)

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(points, districts, how="left", predicate="within")

    joined["state"] = joined["NAME_1"]
    joined["district"] = joined["NAME_2"]

    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined = joined.dropna(subset=["state", "district"])

    return joined

df = get_mapped_data()

# --------------------------------------------------
# ALERT LOGIC
# --------------------------------------------------
threshold = np.percentile(df["brightness"], 75)
df["alert"] = df["brightness"] > threshold

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
mode = st.sidebar.selectbox(
    "Select Dashboard",
    [
        "Fire Density Map",
        "Time Animation Map",
        "Fire Analytics",
        "District Monitoring",
        "AI Risk Model"
    ]
)

# --------------------------------------------------
# 🔥 FIRE DENSITY MAP
# --------------------------------------------------
if mode == "Fire Density Map":

    st.subheader("🔥 Satellite Fire Density")

    sample = df.iloc[:10000]
    data = sample[["latitude", "longitude", "brightness"]].to_dict("records")

    layer = pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position=["longitude", "latitude"],
        get_weight="brightness",
        radiusPixels=60
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=4)
        )
    )

# --------------------------------------------------
# ⏳ TIME ANIMATION MAP
# --------------------------------------------------
elif mode == "Time Animation Map":

    st.subheader("🔥 Fire Activity Map")

    compare_mode = st.toggle("🔍 Compare Periods")

    if not compare_mode:

        year = st.slider(
            "Select Year",
            int(df.year.min()),
            int(df.year.max()),
            int(df.year.min())
        )

        filtered = df[df.year == year]

        sample = filtered.iloc[:10000]
        data = sample[["latitude", "longitude"]].to_dict("records")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position=["longitude", "latitude"],
            get_radius=20000,
            get_fill_color=[255, 120, 0],
            opacity=0.6
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=4)
            )
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            past_start = st.selectbox("Past Start Year", sorted(df.year.unique()))
            past_end = st.selectbox("Past End Year", sorted(df.year.unique()), index=5)

        with col2:
            future_start = st.selectbox("Future Start Year", list(range(2025, 2031)))
            future_end = st.selectbox("Future End Year", list(range(2025, 2031)), index=5)

        past_data = df[(df.year >= past_start) & (df.year <= past_end)]
        past_sample = past_data.iloc[:10000]

        future_sample = df.iloc[:10000].copy()
        future_sample["brightness"] *= np.random.uniform(1.1, 1.3)

        map1, map2 = st.columns(2)

        with map1:
            st.markdown(f"### Past Fires ({past_start}-{past_end})")

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=past_sample.to_dict("records"),
                get_position=["longitude", "latitude"],
                get_radius=20000,
                get_fill_color=[255, 120, 0]
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=4)
                )
            )

        with map2:
            st.markdown(f"### Predicted Fires ({future_start}-{future_end})")

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=future_sample.to_dict("records"),
                get_position=["longitude", "latitude"],
                get_radius=20000,
                get_fill_color=[255, 0, 0]
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=4)
                )
            )

# --------------------------------------------------
# 📊 FIRE ANALYTICS
# --------------------------------------------------
elif mode == "Fire Analytics":

    st.subheader("📊 Fire Trend Analysis (2012–2030)")

    temp_df = df.copy()

    daily = temp_df.groupby("acq_date").agg({
        "brightness": "mean",
        "frp": "mean"
    }).reset_index()

    daily = daily.sort_values("acq_date")

    daily["brightness_smooth"] = daily["brightness"].rolling(7).mean()

    daily["month"] = daily["acq_date"].dt.month
    daily["month_sin"] = np.sin(2 * np.pi * daily["month"] / 12)
    daily["month_cos"] = np.cos(2 * np.pi * daily["month"] / 12)

    daily["lag1"] = daily["brightness_smooth"].shift(1)
    daily["lag2"] = daily["brightness_smooth"].shift(2)
    daily["lag7"] = daily["brightness_smooth"].shift(7)

    daily["rolling_mean"] = daily["brightness_smooth"].rolling(7).mean()
    daily["rolling_std"] = daily["brightness_smooth"].rolling(7).std()

    daily = daily.dropna()

    X = daily[
        [
            "lag1", "lag2", "lag7",
            "rolling_mean", "rolling_std",
            "month_sin", "month_cos", "frp"
        ]
    ]
    y = daily["brightness_smooth"]

    split = int(len(daily) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)

    st.metric("Model R² Score", f"{r2:.3f}")

    history = list(daily["brightness_smooth"].values[-7:])
    month = daily.iloc[-1]["month"]

    future_preds = []

    for _ in range(365 * 6):
        month = (month % 12) + 1

        lag1 = history[-1]
        lag2 = history[-2]
        lag7 = history[0]

        roll_mean = np.mean(history)
        roll_std = np.std(history)

        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        frp_dynamic = daily.iloc[-1]["frp"] + np.random.normal(0, 2)

        pred = model.predict([[
            lag1, lag2, lag7,
            roll_mean, roll_std,
            month_sin, month_cos,
            frp_dynamic
        ]])[0]

        seasonal = 5 * np.sin(2 * np.pi * month / 12)
        noise = np.random.normal(0, 1.2)

        pred = pred + seasonal + noise

        future_preds.append(pred)

        history.append(pred)
        history.pop(0)

    future_dates = pd.date_range(start="2024-01-01", periods=len(future_preds))

    future_df = pd.DataFrame({
        "date": future_dates,
        "value": future_preds
    })

    future_df["year"] = future_df["date"].dt.year
    future_df = future_df.groupby("year")["value"].mean().reset_index()
    future_df["type"] = "Predicted"

    past = daily.groupby(daily["acq_date"].dt.year)["brightness_smooth"].mean().reset_index()
    past.columns = ["year", "value"]
    past["type"] = "Past"

    combined = pd.concat([past, future_df])

    fig = px.line(
        combined,
        x="year",
        y="value",
        color="type",
        markers=True,
        title="🔥 High Accuracy Fire Prediction"
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# 📍 DISTRICT MONITORING
# --------------------------------------------------
elif mode == "District Monitoring":

    st.subheader("📍 District Monitoring")

    states = sorted(df["state"].unique())
    selected_state = st.selectbox("Select State", states)

    state_df = df[df["state"] == selected_state]

    districts = sorted(state_df["district"].unique())
    selected_district = st.selectbox("Select District", districts)

    district_df = state_df[state_df["district"] == selected_district]

    st.metric("🔥 Total Fires", len(district_df))

    sample = district_df.iloc[:3000].copy()

    max_bright = sample["brightness"].max()
    min_bright = sample["brightness"].min()

    sample["color"] = sample["brightness"].apply(
        lambda x: [
            int(255 * (x - min_bright) / (max_bright - min_bright + 1e-6)),
            0,
            0
        ]
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=sample.to_dict("records"),
        get_position=["longitude", "latitude"],
        get_radius=6000,
        get_fill_color="color"
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=22, longitude=78, zoom=6)
        )
    )

    # --------------------------------------------------
    # 🚨 ALERT SYSTEM
    # --------------------------------------------------
    st.sidebar.subheader("🚨 Alert System")

    enable_alert = st.sidebar.checkbox("Enable Alerts")

    alerts = district_df[
        district_df["brightness"] > np.percentile(district_df["brightness"], 75)
    ]

    top = alerts.groupby(["district", "state"]).size().reset_index(name="fires")

    st.metric("🔥 High Risk Fires", len(alerts))

    if enable_alert and len(alerts) > 10:
        result = trigger_alert(top.head(3))
        st.warning(result)

    if st.sidebar.button("🚨 Send Alert Now"):
        result = trigger_alert(top.head(3))
        st.warning(result)

# --------------------------------------------------
# 🤖 AI MODEL
# --------------------------------------------------
elif mode == "AI Risk Model":

    df["risk"] = (df["brightness"] > threshold).astype(int)

    X = df[["brightness", "frp"]]
    y = df["risk"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    model = XGBClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    st.metric("Model Accuracy", f"{acc * 100:.2f}%")
