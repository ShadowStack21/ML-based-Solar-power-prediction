import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@st.cache_data
def generate_solar_data():
    date_range = pd.date_range(start="2023-01-01", end="2023-12-31 23:00:00", freq="h")
    data = pd.DataFrame({"DATE_TIME": date_range})
    
    data["MONTH"] = data["DATE_TIME"].dt.month
    data["HOUR"] = data["DATE_TIME"].dt.hour
    data["WEEK"] = data["DATE_TIME"].dt.isocalendar().week

    def get_season(month):
        if month in [6,7,8]: return "Summer"
        elif month in [9,10,11]: return "Autumn"
        elif month in [12,1,2]: return "Winter"
        else: return "Spring"

    def seasonal_irradiation(month):
        if month in [6,7,8]:       return np.random.uniform(750, 1000) 
        elif month in [9,10,11]:   return np.random.uniform(400, 700)  
        elif month in [12,1,2]:    return np.random.uniform(300, 500)  
        else:                      return np.random.uniform(500, 800)  

    
    def optimal_angle(row):
        season = row["SEASON"]
        hour = row["HOUR"]
        
       
        if season == "Summer": base_tilt = 15
        elif season == "Winter": base_tilt = 45
        else: base_tilt = 30
        
       
        if hour < 6 or hour > 18:
            return 75 
            
        hours_from_noon = abs(hour - 12)
        tilt = base_tilt + (hours_from_noon * 10) 
        return min(tilt, 80) 

    data["SEASON"] = data["MONTH"].apply(get_season)
    data["OPTIMAL_ANGLE"] = data.apply(optimal_angle, axis=1)
    
    
    data["TILT_ANGLE"] = np.random.uniform(0, 80, len(data))
    
    base_irrad = data["MONTH"].apply(seasonal_irradiation)
    
    day_mask = (data["HOUR"] >= 6) & (data["HOUR"] <= 18)
    time_multiplier = np.sin((data["HOUR"] - 6) * np.pi / 12)
    data["IRRADIATION"] = np.where(day_mask, base_irrad * time_multiplier, 0)

    data["AMBIENT_TEMPERATURE"] = np.random.uniform(20, 40, len(data))
    data["MODULE_TEMPERATURE"] = data["AMBIENT_TEMPERATURE"] + np.random.uniform(2, 10, len(data))
    
    
    angle_penalty = 1 - (abs(data["TILT_ANGLE"] - data["OPTIMAL_ANGLE"]) * 0.015)
    angle_penalty = np.clip(angle_penalty, 0.3, 1.0)
    
    ac_power = (data["IRRADIATION"] * 0.6) * angle_penalty + np.random.normal(0, 5, len(data))
    data["AC_POWER"] = np.clip(np.where(day_mask, ac_power, 0), 0, None)

    return data


@st.cache_resource
def train_model(data):
    
    X = data[["MONTH", "HOUR", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TILT_ANGLE"]]
    y = data["AC_POWER"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    return model


st.set_page_config(page_title="AI Solar Optimizer", layout="wide")
st.title("🌤️ AI Solar Power & Hourly Tracker")

data = generate_solar_data()
model = train_model(data)


st.sidebar.header("🎛️ Dashboard Controls")


st.sidebar.subheader("1. Date & Time")
month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
selected_month_name = st.sidebar.selectbox("Current Month", list(month_names.values()), index=5)
in_month = list(month_names.keys())[list(month_names.values()).index(selected_month_name)]

in_hour = st.sidebar.slider("Time of Day", min_value=6, max_value=18, value=12, format="%d:00")


st.sidebar.markdown("---")
st.sidebar.subheader("2. Weather & Panels")
in_irrad = st.sidebar.slider("Irradiation (W/m²)", 0.0, 1000.0, 800.0)
in_amb_temp = st.sidebar.slider("Ambient Temp (°C)", 10.0, 50.0, 25.0)
in_mod_temp = st.sidebar.slider("Module Temp (°C)", 10.0, 60.0, 30.0)
in_angle = st.sidebar.slider("Current Panel Angle (°)", 0, 80, 30)

if st.sidebar.button("Predict Current Output"):
    input_features = pd.DataFrame({"MONTH": [in_month], "HOUR": [in_hour], "IRRADIATION": [in_irrad], "AMBIENT_TEMPERATURE": [in_amb_temp], "MODULE_TEMPERATURE": [in_mod_temp], "TILT_ANGLE": [in_angle]})
    pred = max(0, model.predict(input_features)[0])
    st.sidebar.info(f"Current Output: **{pred:.2f} kW**")

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Auto-Optimization")
if st.sidebar.button("Maximize Power Output"):
    test_angles = np.arange(0, 81, 1)
    df_opt = pd.DataFrame({
        "MONTH": [in_month] * len(test_angles),
        "HOUR": [in_hour] * len(test_angles),
        "IRRADIATION": [in_irrad] * len(test_angles),
        "AMBIENT_TEMPERATURE": [in_amb_temp] * len(test_angles),
        "MODULE_TEMPERATURE": [in_mod_temp] * len(test_angles),
        "TILT_ANGLE": test_angles
    })
    all_preds = model.predict(df_opt)
    best_idx = np.argmax(all_preds)
    
    st.sidebar.success(f"**Optimal Angle at {in_hour}:00 : {test_angles[best_idx]}°**")
    st.sidebar.success(f"**Maximized Power: {all_preds[best_idx]:.2f} kW**")



tab1, tab2, tab3 = st.tabs(["⏱️ Daily Solar Tracker Profile", "📐 Instant Angle Optimizer", "📅 Seasonal Trends"])


with tab1:
    st.subheader(f"Optimal Tracking Profile for {selected_month_name}")
    st.markdown("This shows how an AI-powered Dual-Axis Tracker will move the panels from Morning (6:00) to Evening (18:00) to capture maximum sunlight.")
    
    hours_in_day = np.arange(6, 19, 1)
    best_angles_for_day = []
    
    
    for h in hours_in_day:
        test_angles = np.arange(0, 81, 1)
        df_day = pd.DataFrame({
            "MONTH": [in_month] * len(test_angles),
            "HOUR": [h] * len(test_angles),
            "IRRADIATION": [in_irrad] * len(test_angles), 
            "AMBIENT_TEMPERATURE": [in_amb_temp] * len(test_angles),
            "MODULE_TEMPERATURE": [in_mod_temp] * len(test_angles),
            "TILT_ANGLE": test_angles
        })
        hourly_preds = model.predict(df_day)
        best_angles_for_day.append(test_angles[np.argmax(hourly_preds)])
        
    fig_day, ax_day = plt.subplots(figsize=(10, 4))
    ax_day.plot(hours_in_day, best_angles_for_day, marker='o', color='darkorange', linewidth=3, markersize=8)
    
    
    ax_day.axvline(x=in_hour, color='red', linestyle='--', alpha=0.5, label=f"Current Time ({in_hour}:00)")
    
    ax_day.set_title("AI Automated Panel Movement (Morning to Evening)")
    ax_day.set_xlabel("Time of Day (Hour)")
    ax_day.set_ylabel("Optimal Panel Tilt Angle (°)")
    ax_day.set_xticks(hours_in_day)
    ax_day.invert_yaxis() 
    ax_day.legend()
    ax_day.grid(True, linestyle=':', alpha=0.7)
    st.pyplot(fig_day)


with tab2:
    st.subheader(f"Instant Angle Analysis at {in_hour}:00")
    st.markdown("Change the **Time of Day** in the sidebar. Watch the peak shift steeply in the mornings/evenings, and flatten out at noon!")
    
    test_angles = np.arange(0, 81, 1)
    df_plot = pd.DataFrame({
        "MONTH": [in_month] * len(test_angles),
        "HOUR": [in_hour] * len(test_angles),
        "IRRADIATION": [in_irrad] * len(test_angles),
        "AMBIENT_TEMPERATURE": [in_amb_temp] * len(test_angles),
        "MODULE_TEMPERATURE": [in_mod_temp] * len(test_angles),
        "TILT_ANGLE": test_angles
    })
    curve_preds = model.predict(df_plot)
    best_idx_plot = np.argmax(curve_preds)
    
    fig_opt, ax_opt = plt.subplots(figsize=(10, 4))
    ax_opt.plot(test_angles, curve_preds, color="blue", linewidth=2, label="Power Yield Curve")
    
    current_pred = model.predict([[in_month, in_hour, in_irrad, in_amb_temp, in_mod_temp, in_angle]])[0]
    ax_opt.scatter(in_angle, current_pred, color="red", s=100, zorder=5, label=f"Current Angle ({in_angle}°)")
    ax_opt.scatter(test_angles[best_idx_plot], curve_preds[best_idx_plot], color="green", s=200, marker='*', zorder=5, label=f"Optimal Peak ({test_angles[best_idx_plot]}°)")
    
    ax_opt.set_xlabel("Solar Panel Tilt Angle (Degrees)")
    ax_opt.set_ylabel("Predicted AC Power (kW)")
    ax_opt.grid(True, linestyle='--', alpha=0.6)
    ax_opt.legend()
    st.pyplot(fig_opt)


with tab3:
    st.subheader("Historical Seasonal Trends")
    st.markdown("Because we added `HOUR` tracking, the model is now significantly more accurate across the seasons!")
    
    
    sample_data = data.sample(2000, random_state=42)
    X_sample = sample_data[["MONTH", "HOUR", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "TILT_ANGLE"]]
    sample_data["PREDICTED_AC_POWER"] = np.clip(model.predict(X_sample), 0, None)

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Seasonal Peak Generation**")
        seasonal_peak = sample_data.groupby("SEASON")["AC_POWER"].max()
        fig_s, ax_s = plt.subplots(figsize=(6,4))
        sns.barplot(x=seasonal_peak.index, y=seasonal_peak.values, palette="viridis", ax=ax_s)
        ax_s.set_ylabel("Peak AC Power (kW)")
        st.pyplot(fig_s)

    with col2:
        st.write("**Peak Monthly Generation (Actual vs AI)**")
        monthly_comp = sample_data.groupby("MONTH")[["AC_POWER", "PREDICTED_AC_POWER"]].max()
        fig_m, ax_m = plt.subplots(figsize=(6,4))
        x = np.arange(len(monthly_comp.index))
        width = 0.35
        ax_m.bar(x - width/2, monthly_comp["AC_POWER"], width, label='Actual', color='teal')
        ax_m.bar(x + width/2, monthly_comp["PREDICTED_AC_POWER"], width, label='AI Predicted', color='coral')
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(monthly_comp.index)
        ax_m.legend()
        st.pyplot(fig_m)
