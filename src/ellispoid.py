import numpy as np
import matplotlib.pyplot as plt

# Parameter des Ellipsoids
a = 400
b = 2
c = 1

# Dichtefunktion auf der Oberfläche des Ellipsoids
def p(x):
    return np.sqrt((b * c * np.sin(x[1])**2 * np.cos(x[0])**2 +
                    a * c * np.sin(x[1])**2 * np.sin(x[0])**2 +
                    a * b * np.cos(x[1])**2) / (a * b * c * np.cos(x[1])))

# Akzeptanzfunktion im Metropolis-Algorithmus
def alpha(x, y):
    return min(1, np.sqrt(p(y) / p(x)))

# MCMC-Sampler
T = 10**4
data = np.empty((T, 2))  # Speichervariable für die Punkte
X = np.ones(2) * np.pi / 2  # Anfangspunkt
accept_prob = 0

# Metropolis-Hastings-Schleife
for t in range(T):
    Y = np.array([2 * np.pi * np.random.rand(), np.arccos(1 - 2 * np.random.rand())])  # Vorschlagsverteilung
    if np.random.rand() < alpha(X, Y):  # Akzeptanzprüfung
        X = Y
        accept_prob += 1
    data[t, :] = X

# Akzeptanzrate
accept_prob /= T
print(f"Akzeptanzrate: {accept_prob}")

# Berechnung der Autokorrelationsfunktion
K = 20
x_data = data[:, 0]  # x-Koordinaten der Stichproben
ell = np.mean(x_data)
R = np.zeros(K + 1)

for k in range(K + 1):
    R[k] = np.sum((x_data[:T - k] - ell) * (x_data[k:T] - ell)) / (T - k)

# Normierung auf R[0]
R /= R[0]

# Plot der Autokorrelationsfunktion
plt.figure(figsize=(8, 4))
plt.plot(range(K + 1), R, marker='o')
plt.xlabel('k')
plt.ylabel('R(k)')
plt.title('Autokorrelationsfunktion')
plt.grid(True)
plt.show()

# Umwandlung der sphärischen in kartesische Koordinaten für den Plot
x1 = a * np.cos(data[:, 0]) * np.sin(data[:, 1])
x2 = b * np.sin(data[:, 0]) * np.sin(data[:, 1])
x3 = c * np.cos(data[:, 1])

# 3D-Punktewolke plotten
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, x3, s=1)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Punktewolke auf der Oberfläche des Ellipsoids')
plt.show()
