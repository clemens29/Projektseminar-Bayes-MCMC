import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from sys import argv
import scipy.stats as stats

# Binomialverteilung
def likelihood(p, x, n):
    return (p**x) * ((1 - p)**(n - x))

# Normalverteilung N(0.5, 0.1)
def prior(p):
    return stats.norm.pdf(p, 0.5, 0.1)

# Posteriorverteilung ohne Normalisierungskonstante
def posterior(p, x, n):
    return likelihood(p, x, n) * prior(p)

# Metropolis-Hastings Algorithmus zum Sampling der Posteriorverteilung
def metropolis_hastings(x, n, n_samples, start=0.5, proposal_width=0.1):
    
    samples = [] # Markov-Kette
    current_p = start # Startwert - vorheriger Wert
    
    for _ in range(n_samples):
        
        # Vorschlag für einen neuen Wert aus der Normalverteilung um den vorherigen Wert
        proposed_p = np.random.normal(current_p, proposal_width)
        
        # Ablehnung, wenn der Vorschlag außerhalb des Wertebereichs liegt
        if proposed_p < 0 or proposed_p > 1:
            continue
        
        # Berechnung der Akzeptanzwahrscheinlichkeit
        current_posterior = posterior(current_p, x, n)
        proposed_posterior = posterior(proposed_p, x, n)
        acceptance_ratio = proposed_posterior / current_posterior
        if np.random.rand() < acceptance_ratio:
            current_p = proposed_p  # Akzeptiere den Vorschlag
        
        samples.append(current_p)
    
    return samples[1000:]  # Burn-in entfernen

# Parameter für das Problem
x = int(argv[2])
n = int(argv[1])
sp_p = x / n
n_samples = 100_00

# Sampling aus der Posteriorverteilung
samples = metropolis_hastings(x, n, n_samples, start=0.5, proposal_width=0.1)

#Bayes'sche Schätzung und Konfidenzintervall für p
mean_bayes = np.mean(samples)
konf_bayes_low, konf_bayes_high = np.percentile(samples, [2.5, 97.5])

#Klassische Schätzung und Konfidenzintervall für p
mean_classical = x / n

#sn=x1+x2+...+xn also anzah
sn = x
#f1 - Quantil der F-Verteilung mit 2*sn und 2*(n-sn+1) Freiheitsgraden und 0.025 Wahrscheinlichkeit
f1 = stats.f.ppf(0.025, 2*sn, 2*(n-sn+1))
#f2 - Quantil der F-Verteilung mit 2*(sn+1) und 2*(n-sn) Freiheitsgraden und 0.975 Wahrscheinlichkeit
f2 = stats.f.ppf(0.975, 2*(sn+1), 2*(n-sn))
konf_classical_low = (sn*f1)/(n-sn+1+sn*f1)
konf_classical_high = ((sn+1)*f2)/(n-sn+(sn+1)*f2)

print()
print("----Bayes'sche Schätzung und Konfidenzintervall für p----")
print(f"Bayes'sche Schätzung für p: {mean_bayes:.3f}")
print(f"Konfidenzintervall für p: [{konf_bayes_low:.3f}, {konf_bayes_high:.3f}]")
print()
print("----Klassische Schätzung und Konfidenzintervall für p----")
print(f"Klassische Schätzung für p: {mean_classical:.3f}")
print(f"Konfidenzintervall für p: [{konf_classical_low:.3f}, {konf_classical_high:.3f}]")
print()


# Plotten der berechneten Posteriorverteilung
def norm_binomial(x, n, p):
    coefficient = gamma(n + 2) / (gamma(x + 1) * gamma(n - x + 1))
    return coefficient * (p ** x) * ((1 - p) ** (n - x))

p_values = np.linspace(0, 1, 100)
norm_binom = [norm_binomial(x, n, p) for p in p_values]
priori = [prior(p) for p in p_values]
# Posterior Histogramm und berechnete Binomialverteilung plotten
#plt.plot(p_values, norm_binom, label='Normierte Binomialverteilung', color='orange')
plt.plot(p_values, priori, label='Prior', color='green')
plt.hist(samples, bins=50, density=True, label='Posterior Samples', color='skyblue', edgecolor='black')
plt.xlabel('p (Wahrscheinlichkeit, dass Deutschland ein WM Spiel gewinnt)')
plt.ylabel('Verteilung (nicht normaiert)')
plt.title('Posterior für p')
plt.axvline(np.mean(samples), color='red', linestyle='--', label=f'Erwartungswert: {np.mean(samples):.3f}')
plt.legend()
plt.show()