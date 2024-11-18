import numpy as np
import matplotlib.pyplot as plt

# Definiere die Likelihood für eine Binomialverteilung
def likelihood(p, x, n):
    return (p**x) * ((1 - p)**(n - x))

# Definiere die Prior (Gleichverteilung zwischen 0 und 1)
def prior(p):
    if 0 <= p <= 1:
        return 1
    return 0

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
        
        current_posterior = posterior(current_p, x, n)
        proposed_posterior = posterior(proposed_p, x, n)
        
        if current_posterior > 0 and proposed_posterior >= 0:
            # Berechne Akzeptanzwahrscheinlichkeit
            acceptance_ratio = proposed_posterior / current_posterior
            if np.random.rand() < acceptance_ratio:
                current_p = proposed_p  # Akzeptiere den Vorschlag
        
        samples.append(current_p)
    
    return samples

# Parameter für das Problem
x = 14
n = 20
sp_p = x / n
n_samples = 10000

# Sampling aus der Posteriorverteilung
samples = metropolis_hastings(x, n, n_samples=n_samples, start=0.5, proposal_width=0.1)

# Visualisierung der Ergebnisse
plt.hist(samples, bins=50, density=True, label='Posterior Samples', color='skyblue', edgecolor='black')
plt.xlabel('p (Wahrscheinlichkeit für roten Ball)')
plt.ylabel('Verteilung (nicht normaiert)')
plt.title('Posterior für p')
plt.axvline(np.mean(samples), color='red', linestyle='--', label=f'Erwartungswert: {np.mean(samples):.3f}')
plt.axvline(x=sp_p, color='green', linestyle='--', label='Sp = {}'.format(sp_p))
plt.legend()
plt.show()