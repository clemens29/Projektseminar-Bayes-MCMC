import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sys import argv
import scipy.stats as stats
from matplotlib.animation import FuncAnimation

def likelihood(p: float, x: int, n: int) -> float:
    """
    Berechnet die Likelihood-Funktion für eine Binomialverteilung.
    
    @param p: Wahrscheinlichkeit für einen Erfolg
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    
    @return: Wahrscheinlichkeitswert der Binomialverteilung
    """
    return (p**x) * ((1 - p)**(n - x))

def prior(p: float, samples=None) -> float:
    """
    Berechnet die Prior-Wahrscheinlichkeit unter Annahme einer Normalverteilung mit Mittelwert 0.5 und Standardabweichung 0.1.
    
    @param p: Wahrscheinlichkeit für einen Erfolg
    @param samples: Zusätzliche Parameter für alternative Prior-Berechnungen (optional)
    
    @return: Wahrscheinlichkeitsdichte der Prior-Verteilung
    """
    return stats.norm.pdf(p, 0.5, 0.1)

def posterior(p: float, x: int, n: int) -> float:
    """
    Berechnet die unnormierte Posterior-Wahrscheinlichkeit basierend auf Likelihood und Prior.
    
    @param p: Wahrscheinlichkeit für einen Erfolg
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    
    @return: Unnormierte Posterior-Wahrscheinlichkeit
    """
    return likelihood(p, x, n) * prior(p)

def metropolis_hastings(x: int, n: int, n_samples: int = 10000, start: float = 0.5, proposal_width: float = 0.1, prior=prior, samplesP=None) -> list:
    """
    Fuehrt den Metropolis-Hastings-Algorithmus aus, um Stichproben aus der Posterior-Verteilung zu ziehen.
    
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    @param n_samples: Anzahl der zu ziehenden Stichproben
    @param start: Startwert fuer die Markov-Kette
    @param proposal_width: Standardabweichung der Normalverteilung fuer die Vorschlagswerte
    @param prior: Prior-Funktion zur Berechnung der Wahrscheinlichkeit
    @param samplesP: Zusätzliche Parameter für alternative Prior-Berechnungen
    
    @return: Liste der gezogenen Stichproben
    """
    samples = []  # Liste zur Speicherung der Stichproben (Markov-Kette)
    current_p = start  # Startwert für die Markov-Kette
    
    for _ in range(n_samples):
        # Vorschlag für einen neuen Wert basierend auf einer Normalverteilung um den aktuellen Wert
        proposed_p = np.random.normal(current_p, proposal_width)
        
        # Berechnung der Akzeptanzwahrscheinlichkeit
        current_posterior = prior(current_p, samplesP) * likelihood(current_p, x, n)
        proposed_posterior = prior(proposed_p, samplesP) * likelihood(proposed_p, x, n)
        acceptance_ratio = proposed_posterior / current_posterior
        
        # Akzeptiere den neuen Vorschlag mit der berechneten Wahrscheinlichkeit
        if np.random.rand() < acceptance_ratio:
            current_p = proposed_p  # Aktualisiere den aktuellen Wert
        
        samples.append(current_p)  # Speichere den aktuellen Wert in der Stichprobenliste
    
    return samples

# Parameter für das Problem
x = 72
n = 112
sp_p = x / n
n_samples = 10_000  # Anzahl der Stichproben

def prioriSamples(p: float, samples: list) -> float:
    """
    Berechnet eine empirische Prior-Verteilung auf Basis bisheriger Stichproben.
    
    @param p: Wahrscheinlichkeit fuer einen Erfolg
    @param samples: Liste von Stichproben zur Prior-Berechnung
    
    @return: Empirische Prior-Wahrscheinlichkeit
    """
    delta = 1e-3
    samples = np.asarray(samples)
    close_values = np.abs(samples - p) < delta
    return np.mean(close_values)

# Sampling aus der Posteriorverteilung
samples = metropolis_hastings(x//4, n//4, n_samples=n_samples, start=0.5, proposal_width=0.1)
samplesR = metropolis_hastings(3*x//4, 3*n//4, n_samples=n_samples, start=0.5, proposal_width=0.1, prior=prioriSamples, samplesP=samples)
samplesRT = metropolis_hastings(x, n, n_samples=n_samples, start=0.5, proposal_width=0.1)

# Initialisiere die Werte für p
p_values = np.linspace(0, 1, 100)
priori = [prior(p, None) for p in p_values]

# Maximalen y-Wert für die y-Achse bestimmen
bins = 50
hist1, _ = np.histogram(samples, bins=bins, density=True)
hist2, _ = np.histogram(samplesR, bins=bins, density=True)
hist3, _ = np.histogram(samplesRT, bins=bins, density=True)
y_max = max(hist1.max(), hist2.max(), hist3.max())  # Maximale Höhe aller Histogramme

# Animation erstellen
fig, ax = plt.subplots(figsize=(8, 6))

def update(frame: int):
    """
    Aktualisiert das Histogramm für die Animation.
    
    @param frame: Index der aktuellen Animation (wechselnd zwischen zwei Stichproben)
    """
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