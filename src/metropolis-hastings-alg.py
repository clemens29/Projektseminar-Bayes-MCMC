import numpy as np
import matplotlib.pyplot as plt

# Zielverteilung (z.B. Standard-Normalverteilung)
def target_distribution(x):
    return np.exp(-0.5 * x**2)

# Vorschlagsverteilung (Normalverteilung mit Mittelwert x und Standardabweichung sigma)
def proposal_distribution(x, sigma=1.0):
    return np.random.normal(x, sigma)

# Metropolis-Hastings Algorithmus
def metropolis_hastings(n_samples, burn_in, proposal_sigma=1.0):
    samples = []
    
    # Startwert (beliebiger Punkt)
    x_current = 0.0
    
    # Initialisieren der Kette
    for i in range(n_samples + burn_in):
        x_proposed = proposal_distribution(x_current, proposal_sigma)
        
        # Berechnung der Akzeptanzwahrscheinlichkeit
        acceptance_ratio = min(1, target_distribution(x_proposed) / target_distribution(x_current))
        
        # Entscheiden, ob der neue Punkt akzeptiert wird
        if np.random.rand() < acceptance_ratio:
            x_current = x_proposed
        
        # Speichern des Werts nach Burn-in-Phase
        if i >= burn_in:
            samples.append(x_current)
    
    return np.array(samples)

# Parameter
n_samples = 100000  # Anzahl der Proben nach der Burn-in-Phase
burn_in = 2000     # Burn-in Phase (erste 2000 Schritte werden verworfen)
proposal_sigma = 1.0  # Standardabweichung der Vorschlagsverteilung

# Metropolis-Hastings ausführen
samples = metropolis_hastings(n_samples, burn_in, proposal_sigma)

# Histogramm der Proben anzeigen
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# Plot der Zielverteilung (Standard-Normalverteilung)
x = np.linspace(-4, 4, 1000)
y = target_distribution(x)
plt.plot(x, y, 'r', label="Zielverteilung (Normal)")
plt.legend()
plt.xlabel('x')
plt.ylabel('Dichte')
plt.title('Metropolis-Hastings Simulation')
plt.show()