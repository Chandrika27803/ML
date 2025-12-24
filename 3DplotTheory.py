import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed to activate 3D projection
import numpy as np
print(plt.colormaps())
 
# ----------------------------------------------------
# 1. Create dummy data
# ----------------------------------------------------
# Let's imagine:
# X = Time (seconds)
# Y = Speed (m/s)
# Z = Temperature (°C)
 
# Generate 20 points
x = np.arange(0, 20, 1)           # 0,1,2,...,19 (Time)
y = x * 2                         # Speed (just 2x time for demo)
z = 30 + np.sin(x) * 5            # Temperature around 30°C with small variation
 
# ----------------------------------------------------
# 2. Create 3D figure and axes
# ----------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
 
# ----------------------------------------------------
# 3. Plot data (both line and scatter for clarity)
# ----------------------------------------------------
ax.plot(x, y, z, label="Path", color="blue")        # 3D line
ax.scatter(x, y, z, color="red", s=50)              # 3D points
 
# ----------------------------------------------------
# 4. Label axes and add title
# ----------------------------------------------------
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Speed (m/s)")
ax.set_zlabel("Temperature (°C)")
ax.set_title("3D Demo Plot: Time vs Speed vs Temperature")
 
# Optional: show legend
ax.legend()
 
# Optional: adjust viewing angle
ax.view_init(elev=20, azim=35)  # elevation, azimuth
 
# ----------------------------------------------------
# 5. Show plot
# ----------------------------------------------------
plt.tight_layout()
plt.show()

#Area plot
plt.figure(figsize=(8,5))
plt.fill_between(x,y,color="skyblue",alpha=0.9)
plt.plot(x,y,color="blue",linewidth=2)
plt.title("Area plot Time Vs Speed")
plt.xlabel("Time")
plt.ylabel("speed")
plt.grid(True)
plt.tight_layout()
plt.show()

#3D Area plot
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
X,Y=np.meshgrid(x,y)
Z=np.outer(y,np.ones_like(x))
surf=ax.plot_surface(X,Y,Z,cmap='viridis',alpha=0.5)
ax.set_title("3D area plot")
ax.set_xlabel("X=time")
ax.set_ylabel("Y=speed")
ax.set_zlabel("Z=temp")
plt.tight_layout()
plt.show()

#3D Area plot
fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')
Xg=np.linspace(0,10,40)
Yg=np.linspace(0,10,40)
Xg,Yg=np.meshgrid(Xg,Yg)
Zg=np.sin(Xg)*np.cos(Yg)*10
surf=ax.plot_surface(Xg,Yg,Zg,cmap='terrain',edgecolor="none",alpha=0.5)
ax.set_title("3D area plot")
ax.set_xlabel("X=time")
ax.set_ylabel("Y=speed")
ax.set_zlabel("Z=temp")
ax.view_init(elev=30,azim=45)
plt.tight_layout()
plt.show()
