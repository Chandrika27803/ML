import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

import numpy as np

import seaborn as sns
 
# =====================================================

# 1. LOAD AND PREPARE DATA

# =====================================================

df = pd.read_csv("weather.csv")
 
# Basic checks

print("Columns:", df.columns.tolist())

print(df.head())
 
# Convert types

df["Date"] = pd.to_datetime(df["Date"], errors="raise")

df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

df["SlNo"] = pd.to_numeric(df["SlNo"], errors="coerce")
 
# Drop rows with bad Temperature or SlNo

df = df.dropna(subset=["Temperature", "SlNo"])
 
# Category codes for Condition (for numeric plots & 3D)

df["CondCode"] = df["Condition"].astype("category").cat.codes
 
# =====================================================

# 2. CREATE A LARGE 4x5 GRID OF SUBPLOTS

# =====================================================

fig = plt.figure(figsize=(24, 18))
 
# For convenience

dates = df["Date"].values

temps = df["Temperature"].values

slno = df["SlNo"].values

cond = df["Condition"]

cond_code = df["CondCode"].values
 
# =====================================================

# BASIC PLOTS

# =====================================================
 
# 1. Line plot

ax1 = fig.add_subplot(4, 5, 1)

ax1.plot(dates, temps, marker="o")

ax1.set_title("Line Plot – Temp vs Date")

ax1.set_xlabel("Date")

ax1.set_ylabel("Temp (°C)")

ax1.grid(True)

ax1.tick_params(axis="x", rotation=45)
 
# 2. Scatter plot

ax2 = fig.add_subplot(4, 5, 2)

ax2.scatter(dates, temps, c="red")

ax2.set_title("Scatter Plot")

ax2.set_xlabel("Date")

ax2.set_ylabel("Temp (°C)")

ax2.grid(True)

ax2.tick_params(axis="x", rotation=45)
 
# 3. Bar plot

ax3 = fig.add_subplot(4, 5, 3)

ax3.bar(dates, temps)

ax3.set_title("Bar Plot – Temp per Day")

ax3.set_xlabel("Date")

ax3.set_ylabel("Temp (°C)")

ax3.grid(axis="y")

ax3.tick_params(axis="x", rotation=45)
 
# 4. Histogram

ax4 = fig.add_subplot(4, 5, 4)

ax4.hist(temps, bins=5, edgecolor="black")

ax4.set_title("Histogram – Temp Distribution")

ax4.set_xlabel("Temp (°C)")

ax4.set_ylabel("Frequency")

ax4.grid(True)
 
# 5. Pie chart (Condition frequency)

ax5 = fig.add_subplot(4, 5, 5)

cond_counts = cond.value_counts()

ax5.pie(cond_counts.values, labels=cond_counts.index,

        autopct="%1.1f%%", startangle=90)

ax5.set_title("Pie Chart – Conditions")

ax5.axis("equal")
 
# 6. Box plot

ax6 = fig.add_subplot(4, 5, 6)

ax6.boxplot(temps, vert=True)

ax6.set_title("Box Plot – Temperature")

ax6.set_ylabel("Temp (°C)")

ax6.grid(True)
 
# 7. Area plot

ax7 = fig.add_subplot(4, 5, 7)

ax7.fill_between(dates, temps, alpha=0.4)

ax7.set_title("Area Plot – Temperature")

ax7.set_xlabel("Date")

ax7.set_ylabel("Temp (°C)")

ax7.grid(True)

ax7.tick_params(axis="x", rotation=45)
 
# 8. Step plot

ax8 = fig.add_subplot(4, 5, 8)

ax8.step(dates, temps, where="mid")

ax8.set_title("Step Plot – Temperature")

ax8.set_xlabel("Date")

ax8.set_ylabel("Temp (°C)")

ax8.grid(True)

ax8.tick_params(axis="x", rotation=45)
 
# 9. Stem plot

ax9 = fig.add_subplot(4, 5, 9)

(markerline, stemlines, baseline) = ax9.stem(

    range(len(temps)), temps, use_line_collection=True

)

ax9.set_title("Stem Plot – Index vs Temp")

ax9.set_xlabel("Index")

ax9.set_ylabel("Temp (°C)")

ax9.grid(True)
 
# 10. Horizontal bar (Condition frequency)

ax10 = fig.add_subplot(4, 5, 10)

ax10.barh(cond_counts.index, cond_counts.values)

ax10.set_title("Horizontal Bar – Conditions")

ax10.set_xlabel("Count")

ax10.set_ylabel("Condition")

ax10.grid(axis="x")
 
# =====================================================

# ADVANCED PLOTS

# =====================================================
 
# 11. KDE (density plot) of temperature

ax11 = fig.add_subplot(4, 5, 11)

sns.kdeplot(temps, fill=True, ax=ax11)

ax11.set_title("KDE – Temp Density")

ax11.set_xlabel("Temp (°C)")

ax11.grid(True)
 
# 12. Violin plot for temperature

ax12 = fig.add_subplot(4, 5, 12)

sns.violinplot(y=temps, ax=ax12)

ax12.set_title("Violin Plot – Temp")

ax12.set_ylabel("Temp (°C)")

ax12.grid(True)
 
# 13. Heatmap of correlations

ax13 = fig.add_subplot(4, 5, 13)

corr = df[["SlNo", "Temperature", "CondCode"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax13)

ax13.set_title("Correlation Heatmap")
 
# 14. Dual-axis plot (Temp vs Date, CondCode as bar)

ax14 = fig.add_subplot(4, 5, 14)

ax14.plot(dates, temps, color="blue", marker="o", label="Temp")

ax14.set_xlabel("Date")

ax14.set_ylabel("Temp (°C)", color="blue")

ax14.tick_params(axis="x", rotation=45)

ax14.grid(True)
 
ax14b = ax14.twinx()

ax14b.bar(dates, cond_code, alpha=0.3, color="orange", label="CondCode")

ax14b.set_ylabel("CondCode", color="orange")
 
ax14.set_title("Dual Axis – Temp & Condition Code")
 
# 15. Stacked Area (Temp vs index, plus a shifted version)

ax15 = fig.add_subplot(4, 5, 15)

indices = np.arange(len(temps))

# create a second series just for demo stacking

temps_shifted = temps - temps.min() + 1

ax15.stackplot(indices, temps, temps_shifted, labels=["Temp", "TempShift"])

ax15.set_title("Stacked Area (Demo)")

ax15.set_xlabel("Index")

ax15.set_ylabel("Value")

ax15.legend(loc="upper left")

ax15.grid(True)
 
# =====================================================

# 3D PLOTS

# =====================================================
 
# 16. 3D Scatter (SlNo vs Temp vs CondCode)

ax16 = fig.add_subplot(4, 5, 16, projection="3d")

ax16.scatter(slno, temps, cond_code, c=temps, cmap="viridis")

ax16.set_title("3D Scatter")

ax16.set_xlabel("SlNo")

ax16.set_ylabel("Temp")

ax16.set_zlabel("CondCode")
 
# 17. 3D Line (SlNo vs Temp vs 0)

ax17 = fig.add_subplot(4, 5, 17, projection="3d")

ax17.plot(slno, temps, np.zeros_like(temps))

ax17.set_title("3D Line – SlNo vs Temp")

ax17.set_xlabel("SlNo")

ax17.set_ylabel("Temp")

ax17.set_zlabel("Z=0")
 
# 18. 3D Bar (SlNo vs Temp)

ax18 = fig.add_subplot(4, 5, 18, projection="3d")

xpos = slno

ypos = np.zeros_like(slno)

zpos = np.zeros_like(slno)

dx = np.ones_like(slno) * 0.5

dy = np.ones_like(slno) * 0.5

dz = temps

ax18.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)

ax18.set_title("3D Bar – Temp vs SlNo")

ax18.set_xlabel("SlNo")

ax18.set_ylabel("Y")

ax18.set_zlabel("Temp")
 
# 19. 3D Surface (synthetic surface from temperature)

ax19 = fig.add_subplot(4, 5, 19, projection="3d")

n = len(temps)

X, Y = np.meshgrid(np.arange(n), np.arange(n))

# build a simple surface using outer sum of temps

T = temps - temps.mean()

Z = (T.reshape(-1, 1) + T.reshape(1, -1)) / 2

surf = ax19.plot_surface(X, Y, Z, cmap="plasma")

ax19.set_title("3D Surface (Synthetic from Temp)")

ax19.set_xlabel("X")

ax19.set_ylabel("Y")

ax19.set_zlabel("Value")
 
# 20. Text / Summary panel

ax20 = fig.add_subplot(4, 5, 20)

ax20.axis("off")

text_lines = [

    "Weather Data Overview",

    f"Rows: {len(df)}",

    f"Conditions: {', '.join(cond_counts.index.astype(str))}",

    f"Temp range: {temps.min()} to {temps.max()} °C",

]

ax20.text(0.0, 0.8, "\n".join(text_lines), fontsize=12, transform=ax20.transAxes)
 
# =====================================================

# FINAL

# =====================================================

plt.tight_layout()

plt.show()

 