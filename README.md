# 🌤️ AI Solar Power & Hourly Tracker Optimizer

An advanced Machine Learning web application built using **Streamlit** that simulates seasonal solar energy generation and uses **Random Forest Regression** to predict and optimize solar panel tilt angle for maximum AC power output.

This project models realistic seasonal changes, hourly solar movement, and AI-based automatic panel optimization.

---

## 🚀 Key Features

- ☀️ Seasonal Solar Data Simulation (Spring, Summer, Autumn, Winter)
- ⏱️ Hourly Solar Tracking (6 AM – 6 PM)
- 🤖 Random Forest Regression Model
- ⚡ Real-Time Power Prediction
- 📐 AI-Based Automatic Angle Optimization
- 📊 Daily Optimal Tilt Tracking Profile
- 📅 Seasonal Trend Analysis
- 💾 Streamlit Caching for Performance

---

## 🎯 Project Objective

To develop an AI-powered solar panel optimization system that:

- Predicts AC power output using environmental and panel parameters
- Learns hourly solar movement patterns
- Automatically determines optimal panel tilt angle
- Visualizes seasonal and daily solar trends interactively

---

## 📊 Dataset Description

The dataset is synthetically generated for the full year 2023 with hourly timestamps.

### Included Variables:

- DATE_TIME
- MONTH
- HOUR
- WEEK
- SEASON (Spring, Summer, Autumn, Winter)
- IRRADIATION (W/m²)
- AMBIENT_TEMPERATURE (°C)
- MODULE_TEMPERATURE (°C)
- TILT_ANGLE (Random panel angle)
- OPTIMAL_ANGLE (Physics-based ideal angle)
- AC_POWER (Target variable)

### Data Simulation Includes:

- Seasonal irradiation variation
- Day/Night cycle (6 AM – 6 PM)
- Sinusoidal sunlight intensity pattern
- Temperature variation
- Physics-based tilt angle optimization
- Angle penalty for misalignment
- Noise injection for realism

---

## 🤖 Machine Learning Model

### Model Used:
Random Forest Regressor

### Why Random Forest?

- Captures non-linear relationships
- Handles complex feature interactions
- More powerful than linear regression
- Reduces overfitting through ensemble learning

### Features Used for Training:

- MONTH
- HOUR
- IRRADIATION
- AMBIENT_TEMPERATURE
- MODULE_TEMPERATURE
- TILT_ANGLE

### Target Variable:

- AC_POWER (kW)

Training Strategy:
- 80% Training Data
- 20% Testing Data
- 50 Decision Trees
- Parallel Processing (n_jobs = -1)

---

## 🖥️ Dashboard Overview

The Streamlit app contains three main sections:

### 1️⃣ Daily Solar Tracker Profile
- AI calculates optimal panel angle for every hour (6 AM – 6 PM)
- Visualizes automated panel movement
- Highlights current time position

### 2️⃣ Instant Angle Optimizer
- Shows power yield curve vs tilt angle
- Displays current angle performance
- Identifies optimal tilt for maximum power

### 3️⃣ Seasonal Trends
- Seasonal peak generation comparison
- Monthly peak (Actual vs AI Predicted)
- Demonstrates improved accuracy with hourly tracking

---

## ⚡ AI Auto-Optimization

The system can:

- Predict current power output
- Test all tilt angles from 0° to 80°
- Automatically select the angle that maximizes AC power

This simulates an AI-powered dual-axis solar tracker.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (RandomForestRegressor)

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/ai-solar-optimizer.git
cd ai-solar-optimizer
```

Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

---

## 📈 What Makes This Project Advanced?

- Incorporates physics-inspired solar tilt logic
- Uses ensemble machine learning (Random Forest)
- Learns hourly solar behavior
- Performs AI-driven optimization
- Interactive and real-time system

---

## 🔮 Future Enhancements

- Integration with real solar plant datasets
- Weather API integration
- Dual-axis real-time control simulation
- Deployment on Streamlit Cloud
- Feature importance visualization
- Hyperparameter tuning

---
