import numpy as np
import matplotlib.pyplot as plt

# Wahrer Mittelwert und Standardabweichung
true_mean = 10
true_sd = 2

# Stichprobe von 10 Werten ziehen
np.random.seed(40)
sample_size = 10
data = np.random.normal(loc=true_mean, scale=true_sd, size=sample_size)
print(data)

# Klassische OLS-Schätzung: Mittelwert der Stichprobe
mean_classical = np.mean(data)
print(f"Klassische OLS-Schätzung des Mittelwerts: {mean_classical}")

# Konfidenzintervall berechnen
import scipy.stats as stats

confidence_level = 0.95
sem = stats.sem(data)  # Standardfehler des Mittelwerts
ci = stats.t.interval(confidence_level, len(data)-1, loc=mean_classical, scale=sem)

print(f"95% Konfidenzintervall: {ci}")


# Plot der simulierten Daten
plt.hist(data, bins=5, edgecolor='black', alpha=0.7)
plt.axvline(x=true_mean, color='red', linestyle='--', label=f'True Mean = {true_mean}')
plt.title('Histogram of Sample')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()