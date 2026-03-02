import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
generation = pd.read_csv("Plant_1_Generation_Data.csv")
weather = pd.read_csv("Plant_1_Weather_Sensor_Data.csv")
print("Datasets Loaded Successfully\n")
generation["DATE_TIME"] = pd.to_datetime(generation["DATE_TIME"])
weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"])
merged_data = pd.merge(generation, weather, on="DATE_TIME")
print("Merged Dataset Preview:")
print(merged_data.head())
print("\nAvailable Columns:")
print(merged_data.columns)
X = merged_data[[
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]]
y = merged_data["AC_POWER"]
X = X.dropna()
y = y.loc[X.index]
print("\nMissing Values After Cleaning:")
print(X.isnull().sum())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nTraining Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel Training Completed Successfully!")
