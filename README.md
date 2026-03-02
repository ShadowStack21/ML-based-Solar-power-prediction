# 🌤️ Interactive Seasonal Solar Power Prediction App

An interactive Machine Learning web application built using **Streamlit** that simulates seasonal solar power generation and predicts AC power output using Linear Regression.

This project demonstrates data generation, model training, seasonal analysis, performance evaluation, and real-time prediction in a clean dashboard format.

---

## 🚀 Features

- 📅 Seasonal Solar Data Simulation (Winter, Summer, Monsoon, Post-Monsoon)
- 📊 Weekly, Monthly, and Seasonal Peak Analysis
- 📈 Actual vs Predicted Power Comparison
- 🤖 Linear Regression Model Training
- ⚡ Real-Time AC Power Prediction (User Input)
- 📉 Dynamic Model Performance Metrics (R², MAE, RMSE)
- 🗄️ Interactive Data Filtering by Season
- 💾 Optimized with Streamlit Caching

---

## 🧠 Machine Learning Model

**Algorithm Used:** Linear Regression  
**Target Variable:** AC_POWER  
**Input Features:**
- Irradiation (W/m²)
- Ambient Temperature (°C)
- Module Temperature (°C)

The model is trained on synthetic solar generation data and dynamically evaluates performance based on user-selected seasonal filters.

---

## 📊 How the Data is Generated

- Hourly timestamps for the full year (2023)
- Seasonal irradiation variation
- Day/Night sunlight cycle (6 AM – 6 PM)
- Sinusoidal daylight intensity modeling
- Temperature variations
- Noise injection for realism
- AC power derived from irradiation with linear relation

---

## 🖥️ Dashboard Sections

### 1️⃣ Peak Generation Analysis
- Weekly peak generation
- Seasonal peak comparison
- Monthly peak distribution

### 2️⃣ Trend Comparison
- Weekly actual vs predicted averages
- Monthly peak comparison (grouped bar chart)

### 3️⃣ Model Accuracy (Dynamic)
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Actual vs Predicted scatter visualization

### 4️⃣ Filtered Dataset Viewer
- Displays raw filtered data based on selected seasons

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/solar-power-predictor.git
cd solar-power-predictor
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

## ▶️ Run the App

```bash
streamlit run App.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 🎯 Project Objective

To simulate and predict solar AC power generation using environmental variables, while providing an interactive and analytical dashboard for seasonal energy insights.

---

## 📌 Future Improvements

- Add Random Forest / XGBoost models
- Add real-world solar dataset integration
- Deploy to Streamlit Cloud
- Add feature importance visualization
- Add weather-based simulation enhancements

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

## 👨‍💻 Author

Developed as a Machine Learning and Data Visualization project using Streamlit.
