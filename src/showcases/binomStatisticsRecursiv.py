import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sys import argv
import scipy.stats as stats
from matplotlib.animation import FuncAnimation

# kaggle

# Definiere die Likelihood für eine Binomialverteilung
def likelihood(p, x, n):
    return (p**x) * ((1 - p)**(n - x))

# Definiere die Prior (Gleichverteilung zwischen 0 und 1)
def prior(p, samples):
    return stats.norm.pdf(p, 0.5, 0.1)

# Definiere die Posteriorverteilung (Likelihood * Prior)
def posterior(p, x, n):
    return likelihood(p, x, n) * prior(p)

# Metropolis-Hastings Algorithmus zum Sampling der Posteriorverteilung
def metropolis_hastings(x, n, n_samples=10000, start=0.5, proposal_width=0.1, prior=prior, samplesP=None):
    samples = []
    current_p = start
    for _ in range(n_samples):
        # Vorschlag für einen neuen Wert aus der Normalverteilung
        proposed_p = np.random.normal(current_p, proposal_width)
        
        current_posterior = prior(current_p, samplesP)*likelihood(current_p, x, n)
        proposed_posterior = prior(proposed_p, samplesP)*likelihood(proposed_p, x, n)
        # Berechne Akzeptanzwahrscheinlichkeit
        acceptance_ratio = proposed_posterior / current_posterior
        if np.random.rand() < acceptance_ratio:
            current_p = proposed_p  # Akzeptiere den Vorschlag
        
        samples.append(current_p)
    
    return samples

# Parameter für das Problem
x = 72
n = 112
sp_p = x / n
n_samples = 100_00

def prioriSamples(p,samples):
    delta = 1e-3
    samples = np.asarray(samples)
    close_values = np.abs(samples - p) < delta
    return np.mean(close_values)

# Sampling aus der Posteriorverteilung
samples = metropolis_hastings(x//4, n//4, n_samples=n_samples, start=0.5, proposal_width=0.1)
samplesR = metropolis_hastings(3*x//4, 3*n//4, n_samples=n_samples, start=0.5, proposal_width=0.1, prior=prioriSamples, samplesP=samples)
samplesRT = metropolis_hastings(x, n, n_samples=n_samples, start=0.5, proposal_width=0.1)

p_values = np.linspace(0, 1, 100)
priori = [prior(p,None) for p in p_values]

# Maximalen y-Wert für die y-Achse bestimmen
bins = 50
hist1, _ = np.histogram(samples, bins=bins, density=True)
hist2, _ = np.histogram(samplesR, bins=bins, density=True)
hist3, _ = np.histogram(samplesRT, bins=bins, density=True)
y_max = max(hist1.max(), hist2.max(), hist3.max())  # Maximale Höhe aller Histogramme

# Animation erstellen
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame):
    ax.clear()
    if frame % 2 == 0:
        data = samplesR
        color = 'red'
    else:
        data = samplesRT
        color = 'green'
    
    # Histogramme zeichnen
    ax.hist(samples, bins=bins, density=True, label='Posterior 1', color='skyblue', edgecolor='black', alpha=0.5)
    ax.hist(data, bins=bins, density=True, label='Posterior 2', color=color, edgecolor='black', alpha=0.7)
    ax.plot(p_values, priori, label='Prior', color='green')
    
    # Achsentitel und Details
    ax.set_xlabel('p (Wahrscheinlichkeit, dass Deutschland ein WM Spiel gewinnt)')
    ax.set_ylabel('Verteilung')
    ax.set_title('Posterior für p')
    ax.legend()
    
    # y-Achse fixieren
    ax.set_ylim(0, y_max)

# Animation konfigurieren
ani = FuncAnimation(fig, update, frames=20, interval=500)  # 20 Frames, 500 ms Pause zwischen Frames

# Animation anzeigen
plt.show()