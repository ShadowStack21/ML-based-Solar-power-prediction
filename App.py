import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==========================================
# 1. SYNTHETIC DATA GENERATION 
# ==========================================
@st.cache_data
def generate_solar_data():
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="h")
    data = pd.DataFrame({"DATE_TIME": date_range})
    
    data["MONTH"] = data["DATE_TIME"].dt.month
    data["HOUR"] = data["DATE_TIME"].dt.hour
    data["WEEK"] = data["DATE_TIME"].dt.isocalendar().week

    def seasonal_irradiation(month):
        if month in [3,4,5,6]:     return np.random.uniform(750, 1000) # Summer
        elif month in [7,8,9]:     return np.random.uniform(300, 600)  # Monsoon
        elif month in [10,11]:     return np.random.uniform(500, 800)  # Post-Monsoon
        else:                      return np.random.uniform(400, 700)  # Winter

    def get_season(month):
        if month in [3,4,5,6]: return "Summer"
        elif month in [7,8,9]: return "Monsoon"
        elif month in [10,11]: return "Post-Monsoon"
        else: return "Winter"

    data["SEASON"] = data["MONTH"].apply(get_season)
    
    # Generate Base Irradiation
    base_irrad = data["MONTH"].apply(seasonal_irradiation)
    
    # Apply Day/Night Cycle (Sunlight only between 6 AM and 6 PM)
    day_mask = (data["HOUR"] >= 6) & (data["HOUR"] <= 18)
    time_multiplier = np.sin((data["HOUR"] - 6) * np.pi / 12)
    data["IRRADIATION"] = np.where(day_mask, base_irrad * time_multiplier, 0)

    data["AMBIENT_TEMPERATURE"] = np.random.uniform(20, 40, len(data))
    data["MODULE_TEMPERATURE"] = data["AMBIENT_TEMPERATURE"] + np.random.uniform(2, 10, len(data))
    
    # AC Power generation with some noise (0 at night)
    ac_power = (data["IRRADIATION"] * 0.6) + np.random.normal(0, 5, len(data))
    data["AC_POWER"] = np.clip(np.where(day_mask, ac_power, 0), 0, None)

    return data

# ==========================================
# 2. MODEL TRAINING (Global Model)
# ==========================================
@st.cache_resource
def train_model(data):
    X = data[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]]
    y = data["AC_POWER"]

    # Train model on the entire dataset to keep it robust
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

# ==========================================
# 3. STREAMLIT UI & INTERACTIVITY
# ==========================================
st.set_page_config(page_title="Interactive Solar Predictor", layout="wide")
st.title("🌤️ Interactive Seasonal Solar Power Analysis")

# Load Data and Train Global Model
data = generate_solar_data()
model = train_model(data)

# --- SIDEBAR UI ---
st.sidebar.header("🎛️ User Controls")

# 1. Interactive Season Filter
st.sidebar.subheader("1. Filter Dashboard by Season")
seasons_list = data["SEASON"].unique().tolist()
selected_seasons = st.sidebar.multiselect(
    "Choose Seasons to view:",
    options=seasons_list,
    default=seasons_list # Default shows all seasons
)

# Filter the dataset based on user selection
if len(selected_seasons) == 0:
    st.warning("⚠️ Please select at least one season from the sidebar to view data.")
    st.stop() # Stops execution until a season is selected

# Use .copy() to avoid Pandas SettingWithCopy warnings
filtered_data = data[data["SEASON"].isin(selected_seasons)].copy()

st.sidebar.markdown("---")

# 2. Real-Time Predictor (WITH BUTTON)
st.sidebar.subheader("2. Real-Time Prediction")
in_irrad = st.sidebar.slider("Irradiation (W/m²)", 0.0, 1000.0, 500.0)
in_amb_temp = st.sidebar.slider("Ambient Temp (°C)", 10.0, 50.0, 25.0)
in_mod_temp = st.sidebar.slider("Module Temp (°C)", 10.0, 60.0, 30.0)

# Prediction executes ONLY when button is clicked
if st.sidebar.button("Predict AC Power"):
    input_features = pd.DataFrame({
        "IRRADIATION": [in_irrad], 
        "AMBIENT_TEMPERATURE": [in_amb_temp], 
        "MODULE_TEMPERATURE": [in_mod_temp]
    })
    pred = model.predict(input_features)[0]
    pred = max(0, pred) # Prevent negative predictions
    st.sidebar.success(f"⚡ **Predicted AC Power: {pred:.2f} kW**")

# --- MAIN DASHBOARD TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["📅 Peak Generation Analysis", "📊 Trend Comparison", "📈 Model Accuracy (Dynamic)", "🗄️ Filtered Dataset"])

with tab1:
    st.subheader(f"Solar Peaks for: {', '.join(selected_seasons)}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekly Peak
        st.write("**Weekly Peak Solar Generation**")
        weekly_peak = filtered_data.groupby("WEEK")["AC_POWER"].max()
        fig_week, ax_week = plt.subplots(figsize=(6,4))
        weekly_peak.plot(ax=ax_week, color='coral', marker='o')
        ax_week.set_xlabel("Week Number")
        ax_week.set_ylabel("Peak AC Power (kW)")
        ax_week.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_week)

    with col2:
        # Seasonal Peak 
        st.write("**Seasonal Peak Solar Generation**")
        seasonal_peak = filtered_data.groupby("SEASON")["AC_POWER"].max()
        fig_season, ax_season = plt.subplots(figsize=(6,4))
        sns.barplot(x=seasonal_peak.index, y=seasonal_peak.values, palette="viridis", ax=ax_season)
        ax_season.set_xlabel("Season")
        ax_season.set_ylabel("Peak AC Power (kW)")
        st.pyplot(fig_season)
        
    # Monthly Peak
    st.write("**Monthly Peak Solar Generation**")
    monthly_peak = filtered_data.groupby("MONTH")["AC_POWER"].max()
    fig_month, ax_month = plt.subplots(figsize=(10,3))
    monthly_peak.plot(ax=ax_month, color='teal', kind='bar')
    ax_month.set_xlabel("Month")
    ax_month.set_ylabel("Peak AC Power (kW)")
    plt.xticks(rotation=0)
    st.pyplot(fig_month)

with tab2:
    st.subheader(f"Actual vs Predicted Comparison for: {', '.join(selected_seasons)}")
    
    # Generate predictions for the filtered dataset
    X_filtered = filtered_data[["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]]
    filtered_data["PREDICTED_AC_POWER"] = np.clip(model.predict(X_filtered), 0, None)

    # 1. WEEKLY COMPARISON (Line Chart)
    st.write("**Average Weekly Generation (Actual vs Predicted)**")
    weekly_comp = filtered_data.groupby("WEEK")[["AC_POWER", "PREDICTED_AC_POWER"]].mean()
    
    fig_w_comp, ax_w_comp = plt.subplots(figsize=(10, 4))
    ax_w_comp.plot(weekly_comp.index, weekly_comp["AC_POWER"], label="Actual Power", color="blue", marker="o", linewidth=2)
    ax_w_comp.plot(weekly_comp.index, weekly_comp["PREDICTED_AC_POWER"], label="Predicted Power", color="orange", marker="x", linestyle="--", linewidth=2)
    ax_w_comp.set_xlabel("Week Number")
    ax_w_comp.set_ylabel("Average AC Power (kW)")
    ax_w_comp.legend()
    ax_w_comp.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig_w_comp)

    # 2. MONTHLY COMPARISON (Grouped Bar Chart)
    st.write("**Peak Monthly Generation (Actual vs Predicted)**")
    monthly_comp = filtered_data.groupby("MONTH")[["AC_POWER", "PREDICTED_AC_POWER"]].max()
    
    fig_m_comp, ax_m_comp = plt.subplots(figsize=(10, 4))
    x = np.arange(len(monthly_comp.index))  # label locations
    width = 0.35  # width of the bars

    # Plot grouped bars
    ax_m_comp.bar(x - width/2, monthly_comp["AC_POWER"], width, label='Actual Peak', color='teal')
    ax_m_comp.bar(x + width/2, monthly_comp["PREDICTED_AC_POWER"], width, label='Predicted Peak', color='coral')

    ax_m_comp.set_xlabel('Month')
    ax_m_comp.set_ylabel('Peak AC Power (kW)')
    ax_m_comp.set_xticks(x)
    ax_m_comp.set_xticklabels(monthly_comp.index)
    ax_m_comp.legend()
    st.pyplot(fig_m_comp)

with tab3:
    st.subheader(f"Model Performance during {', '.join(selected_seasons)}")
    
    y_filtered_actual = filtered_data["AC_POWER"]
    y_filtered_pred = filtered_data["PREDICTED_AC_POWER"]
    
    r2_dynamic = r2_score(y_filtered_actual, y_filtered_pred)
    mae_dynamic = mean_absolute_error(y_filtered_actual, y_filtered_pred)
    rmse_dynamic = np.sqrt(mean_squared_error(y_filtered_actual, y_filtered_pred))
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Dynamic R² Score", f"{r2_dynamic:.4f}")
    c2.metric("Dynamic MAE", f"{mae_dynamic:.2f} kW")
    c3.metric("Dynamic RMSE", f"{rmse_dynamic:.2f} kW")
    
    st.markdown("---")
    
    # Actual vs Predicted Plot
    st.write("**Actual vs Predicted Power Output (Filtered Data)**")
    fig_pred, ax_pred = plt.subplots(figsize=(8, 4))
    ax_pred.scatter(y_filtered_actual, y_filtered_pred, alpha=0.4, color='royalblue')
    ax_pred.plot([y_filtered_actual.min(), y_filtered_actual.max()], 
                 [y_filtered_actual.min(), y_filtered_actual.max()], 'r--', lw=2)
    ax_pred.set_xlabel("Actual AC Power (kW)")
    ax_pred.set_ylabel("Predicted AC Power (kW)")
    st.pyplot(fig_pred)

with tab4:
    st.subheader(f"Raw Data Viewer ({len(filtered_data)} records found)")
    # View the data that matches the user's selected seasons
    st.dataframe(filtered_data.reset_index(drop=True), use_container_width=True)
