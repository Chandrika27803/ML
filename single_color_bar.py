import pandas as pd

import matplotlib.pyplot as plt
 
df = pd.read_csv("weather.csv")

df["Date"] = pd.to_datetime(df["Date"])

df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
 
plt.figure(figsize=(10,5))

plt.bar(df["Date"], df["Temperature"], color="skyblue")
 
plt.title("Temperature Over Time")

plt.xlabel("Date")

plt.ylabel("Temperature (°C)")

plt.xticks(rotation=45)

plt.grid(axis="y")

plt.tight_layout()

plt.show()

 