import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
 
df = pd.read_csv("weather.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
 
# Normalize temperature 0–1 for colormap

norm = (df["Temperature"] - df["Temperature"].min()) / (df["Temperature"].max() - df["Temperature"].min())
 
plt.figure(figsize=(10,5))

plt.bar(df["Date"], df["Temperature"], color=plt.cm.coolwarm(norm))
 
plt.title("Temperature Over Time (Gradient Colored)")

plt.xlabel("Date")

plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.show()

 