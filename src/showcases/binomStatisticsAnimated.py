import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sys import argv
import scipy.stats as stats
from matplotlib.animation import FuncAnimation

# Definiere die Likelihood für eine Binomialverteilung
def likelihood(p, x, n):
    return (p**x) * ((1 - p)**(n - x))

# Definiere die Prior (Gleichverteilung zwischen 0 und 1)
def prior(p):
    return stats.norm.pdf(p, 0.5, 0.1)

# Definiere die Posteriorverteilung (Likelihood * Prior)
def posterior(p, x, n):
    return likelihood(p, x, n) * prior(p)

# Metropolis-Hastings Algorithmus zum Sampling der Posteriorverteilung
def metropolis_hastings(x, n, n_samples=10000, start=0.5, proposal_width=0.1):
    samples = []
    current_p = start
    for _ in range(n_samples):
        # Vorschlag für einen neuen Wert aus der Normalverteilung
        proposed_p = np.random.normal(current_p, proposal_width)
        if proposed_p < 0 or proposed_p > 1:
            continue
        
        current_posterior = posterior(current_p, x, n)
        proposed_posterior = posterior(proposed_p, x, n)
        # Berechne Akzeptanzwahrscheinlichkeit
        acceptance_ratio = proposed_posterior / current_posterior
        if np.random.rand() < acceptance_ratio:
            current_p = proposed_p  # Akzeptiere den Vorschlag
        
        samples.append(current_p)
    
    return samples

# Parameter für das Problem
sp_p = float(argv[1])
n_samples = 10_000

p_values = np.linspace(0, 1, 100)
priori = [prior(p) for p in p_values]

n_values = [2, 10, 30, 50, 100, 200]
posterior_data = []
for n in n_values:
    x = int(sp_p * n)
    samples = metropolis_hastings(x, n, n_samples=n_samples, start=0.5, proposal_width=0.1)
    posterior_data.append(samples)

fig, ax = plt.subplots()
bins = np.linspace(0, 1, 100)
priori = [prior(p) for p in p_values]
ax.plot(p_values, priori, label='Prior', color='green')

# Initiales Histogramm
hist_data, _ = np.histogram(posterior_data[0], bins=bins, density=True)
bars = ax.bar(bins[:-1], hist_data, width=np.diff(bins), color='skyblue', edgecolor='black')

# Achsentitel und Layout
ax.set_xlabel('p (Wahrscheinlichkeit, dass Deutschland ein WM Spiel gewinnt)')
ax.set_ylabel('Verteilung (normiert)')
ax.set_title('Posterior für p')
legend = ax.legend(loc='upper left')
ax.set_ylim(0, max(np.histogram(posterior_data[-1], bins=bins, density=True)[0]) * 1.1)  # 10% Puffer für bessere Darstellung
# Update-Funktion für die Animation
def update(frame):
    # Daten für das aktuelle Histogramm
    hist_data, _ = np.histogram(posterior_data[frame], bins=bins, density=True)
    # Aktualisiere die Höhe der Balken
    for bar, h in zip(bars, hist_data):
        bar.set_height(h)
# Animation erstellen
ani = FuncAnimation(fig, update, frames=len(n_values), interval=1000, repeat=True)
# Speichern der Animation
ani.save('binomStatisticsAnimated.gif', writer='ffmpeg', fps=1)

# Animation anzeigen
plt.show()