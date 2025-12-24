import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
 
df = pd.read_csv("weather.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
 
# Generate random colors (RGB)

colors = np.random.rand(len(df), 3)
 
plt.figure(figsize=(10,5))

plt.bar(df["Date"], df["Temperature"], color=colors)
 
plt.title("Temperature Over Time (Random Colors)")

plt.xlabel("Date")

plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.show()

 