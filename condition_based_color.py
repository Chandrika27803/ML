import pandas as pd
import matplotlib.pyplot as plt
 
# ------------------------------
# 1. Read Data
# ------------------------------
df = pd.read_csv("weather.csv")
df["Date"] = pd.to_datetime(df["Date"])
df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
 
# ------------------------------
# 2. Define Condition → Color Map
# ------------------------------
condition_colors = {
    "Sunny": "gold",
    "Rainy": "dodgerblue",
    "Cloudy": "gray",
    "Windy": "lightgreen",
    "Stormy": "red",
}
 
# Map colors safely (default to black if unknown)
df["Color"] = df["Condition"].map(condition_colors).fillna("black")
 
# ------------------------------
# 3. Bar Chart
# ------------------------------
plt.figure(figsize=(10,5))
plt.bar(df["Date"], df["Temperature"], color=df["Color"])
 
plt.title("Temperature Over Time (Colored by Condition)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.xticks(rotation=45)
plt.grid(axis="y")
plt.tight_layout()
plt.show()