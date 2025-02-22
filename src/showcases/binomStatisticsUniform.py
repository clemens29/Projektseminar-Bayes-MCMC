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

def prior(p: float) -> float:
    """
    Berechnet die Prior-Wahrscheinlichkeit unter Annahme einer Gleichverteilung im Intervall [0,1].
    
    @param p: Wahrscheinlichkeit für einen Erfolg
    
    @return: Wahrscheinlichkeitsdichte der Prior-Verteilung
    """
    return 1 if 0 <= p <= 1 else 0

def posterior(p: float, x: int, n: int) -> float:
    """
    Berechnet die unnormierte Posterior-Wahrscheinlichkeit basierend auf Likelihood und Prior.
    
    @param p: Wahrscheinlichkeit für einen Erfolg
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    
    @return: Unnormierte Posterior-Wahrscheinlichkeit
    """
    return likelihood(p, x, n) * prior(p)

def metropolis_hastings(x: int, n: int, n_samples: int = 10000, start: float = 0.5, proposal_width: float = 0.1) -> list:
    """
    Führt den Metropolis-Hastings-Algorithmus aus, um Stichproben aus der Posterior-Verteilung zu ziehen.
    
    @param x: Anzahl der Erfolge (z.B. gewonnene Spiele)
    @param n: Gesamtanzahl der Versuche (z.B. gespielte Spiele)
    @param n_samples: Anzahl der zu ziehenden Stichproben
    @param start: Startwert fuer die Markov-Kette
    @param proposal_width: Standardabweichung der Normalverteilung fuer die Vorschlagswerte
    
    @return: Liste der gezogenen Stichproben
    """
    samples = []  # Liste zur Speicherung der Stichproben (Markov-Kette)
    current_p = start  # Startwert fuer die Markov-Kette
    
    for _ in range(n_samples):
        
        # Vorschlag fuer einen neuen Wert basierend auf einer Normalverteilung um den aktuellen Wert
        proposed_p = np.random.normal(current_p, proposal_width)
        
        # Ablehnung des Vorschlags, falls er ausserhalb des Wertebereichs [0,1] liegt
        if proposed_p < 0 or proposed_p > 1:
            continue
        
        # Berechnung der Akzeptanzwahrscheinlichkeit
        current_posterior = posterior(current_p, x, n)
        proposed_posterior = posterior(proposed_p, x, n)
        acceptance_ratio = proposed_posterior / current_posterior
        
        # Akzeptiere den neuen Vorschlag mit der berechneten Wahrscheinlichkeit
        if np.random.rand() < acceptance_ratio:
            current_p = proposed_p  # Aktualisiere den aktuellen Wert
        
        samples.append(current_p)  # Speichere den aktuellen Wert in der Stichprobenliste
    
    return samples

# Einlesen des wahren Erfolgsanteils aus der Kommandozeile
sp_p = float(argv[1])  # Wahrer Erfolgsanteil
n_samples = 10_000  # Anzahl der Stichproben

# Initialisiere die Werte für p
p_values = np.linspace(0, 1, 100)
priori = [prior(p) for p in p_values]

# Definiere verschiedene Stichprobengrößen
n_values = [2, 10, 30, 50, 100, 200]
posterior_data = []
for n in n_values:
    x = int(sp_p * n)  # Berechnung der Anzahl der Erfolge basierend auf der wahren Erfolgswahrscheinlichkeit
    samples = metropolis_hastings(x, n, n_samples=n_samples, start=0.5, proposal_width=0.1)
    posterior_data.append(samples)

# Initialisiere das Diagramm
fig, ax = plt.subplots()
bins = np.linspace(0, 1, 100)
ax.plot(p_values, priori, label='Prior', color='green')

# Initialisiere das Histogramm
hist_data, _ = np.histogram(posterior_data[0], bins=bins, density=True)
bars = ax.bar(bins[:-1], hist_data, width=np.diff(bins), color='skyblue', edgecolor='black')

# Achsentitel und Layout
ax.set_xlabel('p (Wahrscheinlichkeit, dass Deutschland ein WM Spiel gewinnt)')
ax.set_ylabel('Verteilung (normiert)')
ax.set_title('Posterior für p')
legend = ax.legend(loc='upper left')
ax.set_ylim(0, max(np.histogram(posterior_data[-1], bins=bins, density=True)[0]) * 1.1)  # 10% Puffer für bessere Darstellung

def update(frame):
    """
    Aktualisiert das Histogramm für die Animation.
    
    @param frame: Index der aktuellen Animation (entspricht einer Stichprobengröße in n_values)
    """
    hist_data, _ = np.histogram(posterior_data[frame], bins=bins, density=True)
    for bar, h in zip(bars, hist_data):
        bar.set_height(h)

# Erstelle die Animation
ani = FuncAnimation(fig, update, frames=len(n_values), interval=1000, repeat=True)

# Speichern der Animation als GIF
ani.save('binomStatisticsAnimated.gif', writer='ffmpeg', fps=1)

# Animation anzeigen
plt.show()