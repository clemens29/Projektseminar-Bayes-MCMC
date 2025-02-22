import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sys import argv
import scipy.stats as stats

def likelihood(p: float, x: int, n: int) -> float:
    """
    Berechnet die Likelihood-Funktion fuer eine Binomialverteilung.
    
    @param p: Wahrscheinlichkeit fuer einen Erfolg
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    
    @return: Wahrscheinlichkeitswert der Binomialverteilung
    """
    return (p**x) * ((1 - p)**(n - x))

def prior(p: float) -> float:
    """
    Berechnet die Prior-Wahrscheinlichkeit unter Annahme einer Normalverteilung mit Mittelwert 0.5 und Standardabweichung 0.1.
    
    @param p: Wahrscheinlichkeit fuer einen Erfolg
    
    @return: Wahrscheinlichkeitsdichte der Prior-Verteilung
    """
    return stats.norm.pdf(p, 0.5, 0.1)

def posterior(p: float, x: int, n: int) -> float:
    """
    Berechnet die unnormierte Posterior-Wahrscheinlichkeit basierend auf Likelihood und Prior.
    
    @param p: Wahrscheinlichkeit fuer einen Erfolg
    @param x: Anzahl der Erfolge
    @param n: Gesamtanzahl der Versuche
    
    @return: Unnormierte Posterior-Wahrscheinlichkeit
    """
    return likelihood(p, x, n) * prior(p)

def metropolis_hastings(x: int, n: int, n_samples: int, start: float = 0.5, proposal_width: float = 0.1) -> list:
    """
    Fuehrt den Metropolis-Hastings-Algorithmus aus, um Stichproben aus der Posterior-Verteilung zu ziehen.
    
    @param x: Anzahl der Erfolge (z.B. gewonnene Spiele)
    @param n: Gesamtanzahl der Versuche (z.B. gespielte Spiele)
    @param n_samples: Anzahl der zu ziehenden Stichproben
    @param start: Startwert fuer die Markov-Kette
    @param proposal_width: Standardabweichung der Normalverteilung fuer die Vorschlagswerte
    
    @return: Liste der gezogenen Stichproben (ohne Burn-in-Phase)
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
    
    return samples[1000:]  # Entferne die ersten 1000 Werte (Burn-in-Phase)

# Einlesen der Parameter von der Kommandozeile
x = int(argv[2])  # Anzahl der Erfolge (z.B. gewonnene Spiele)
n = int(argv[1])  # Gesamtanzahl der Versuche (z.B. gespielte Spiele)
sp_p = x / n  # Haeufigkeitsbasierte Schaetzung von p
n_samples = 100_00  # Anzahl der Stichproben aus der Posterior-Verteilung

# Ziehung von Stichproben aus der Posterior-Verteilung
samples = metropolis_hastings(x, n, n_samples, start=0.5, proposal_width=0.1)

# Berechnung der Bayes'schen Schaetzung und des 95%-Konfidenzintervalls
mean_bayes = np.mean(samples)
konf_bayes_low, konf_bayes_high = np.percentile(samples, [2.5, 97.5])

# Berechnung der klassischen Schaetzung und des Konfidenzintervalls auf Basis der F-Verteilung
mean_classical = x / n
sn = x
f1 = stats.f.ppf(0.025, 2*sn, 2*(n-sn+1))
f2 = stats.f.ppf(0.975, 2*(sn+1), 2*(n-sn))
konf_classical_low = (sn*f1)/(n-sn+1+sn*f1)
konf_classical_high = ((sn+1)*f2)/(n-sn+(sn+1)*f2)

# Ausgabe der Ergebnisse
print()
print("----Bayes'sche Schätzung und Konfidenzintervall für p----")
print(f"Bayes'sche Schätzung für p: {mean_bayes:.3f}")
print(f"Konfidenzintervall für p: [{konf_bayes_low:.3f}, {konf_bayes_high:.3f}]")
print()
print("----Klassische Schätzung und Konfidenzintervall für p----")
print(f"Klassische Schätzung für p: {mean_classical:.3f}")
print(f"Konfidenzintervall für p: [{konf_classical_low:.3f}, {konf_classical_high:.3f}]")
print()

# Visualisierung der Prior- und Posterior-Verteilung
p_values = np.linspace(0, 1, 100)
priori = [prior(p) for p in p_values]
plt.plot(p_values, priori, label='Prior', color='green')
plt.hist(samples, bins=50, density=True, label='Posterior Samples', color='skyblue', edgecolor='black')
plt.xlabel('p (Wahrscheinlichkeit, dass Deutschland ein WM Spiel gewinnt)')
plt.ylabel('Verteilung')
plt.title('Posterior für p')
plt.axvline(np.mean(samples), color='red', linestyle='--', label=f'Erwartungswert: {np.mean(samples):.3f}')
plt.legend()
plt.show()