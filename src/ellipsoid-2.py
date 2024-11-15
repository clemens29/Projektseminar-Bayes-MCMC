import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameter des Ellipsoids
a, b, c = 400, 2, 1

# Anzahl der zufälligen Punkte, die wir generieren wollen
num_points = 10000

# Erzeugen zufälliger Punkte innerhalb des umgebenden Würfels
x = np.random.uniform(-a, a, num_points)
y = np.random.uniform(-b, b, num_points)
z = np.random.uniform(-c, c, num_points)

# Bedingung für Punkte auf der Ellipsoid-Oberfläche
# Ein Bereich nahe 1 wird erlaubt, um "auf der Oberfläche" zu akzeptieren
threshold = 0.01
ellipsoid_condition = np.abs((x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) - 1) < threshold

# Punkte auswählen, die die Ellipsoid-Gleichung (ungefähr) erfüllen
x_on_ellipsoid = x[ellipsoid_condition]
y_on_ellipsoid = y[ellipsoid_condition]
z_on_ellipsoid = z[ellipsoid_condition]

# Plotten der Punkte, die auf dem Ellipsoid liegen
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_on_ellipsoid, y_on_ellipsoid, z_on_ellipsoid, color='blue', s=1)

# Achsenlimits einstellen
ax.set_xlim([-a, a])
ax.set_ylim([-b, b])
ax.set_zlim([-c, c])

# Achsen beschriften
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_title("Zufällige Punkte auf der Oberfläche des Ellipsoids")

plt.show()