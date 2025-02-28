import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# Definieren der Prior-Parameter für die Beta-Verteilungen
alpha_E = 37
beta_E = 30
alpha_neg_E = 37
beta_neg_E = 12

# Parameter für die gesamte Siegquote (Likelihood)
alpha_total = 73
beta_total = 41

# Berechnen der Posterior-Parameter
alpha_post_E = alpha_E + alpha_total
beta_post_E = beta_E + beta_total
alpha_post_neg_E = alpha_neg_E + alpha_total
beta_post_neg_E = beta_neg_E + beta_total
print(f"Posterior-Parameter für europäische Teams: alpha={alpha_post_E}, beta={beta_post_E}")
print(f"Posterior-Parameter für nicht-europäische Teams: alpha={alpha_post_neg_E}, beta={beta_post_neg_E}")
print()

# Erwartungswert der Posterior-Verteilungen
E_post_E = alpha_post_E / (alpha_post_E + beta_post_E)
E_post_neg_E = alpha_post_neg_E / (alpha_post_neg_E + beta_post_neg_E)

# Berechnen des 95%-Konfidenzintervalls
CI_post_E = beta.ppf([0.025, 0.975], alpha_post_E, beta_post_E)
CI_post_neg_E = beta.ppf([0.025, 0.975], alpha_post_neg_E, beta_post_neg_E)

# Ausgabe der Erwartungswerte und Konfidenzintervalle
print(f"Erwartungswert für europäische Teams (π_E): {E_post_E:.4f}")
print(f"Erwartungswert für nicht-europäische Teams (π_¬E): {E_post_neg_E:.4f}")
print()
print(f"95%-Konfidenzintervall für π_E: [{CI_post_E[0]:.4f}, {CI_post_E[1]:.4f}]")
print(f"95%-Konfidenzintervall für π_¬E: [{CI_post_neg_E[0]:.4f}, {CI_post_neg_E[1]:.4f}]")
print()

# Plot für Prior-, Likelihood- und Posterior-Verteilungen
p = np.linspace(0, 1, 1000)
prior_E = beta.pdf(p, alpha_E, beta_E)
prior_neg_E = beta.pdf(p, alpha_neg_E, beta_neg_E)
likelihood_G = beta.pdf(p, alpha_total, beta_total)

# Posterior für europäische und nicht-europäische Teams
posterior_E = beta.pdf(p, alpha_post_E, beta_post_E)
posterior_neg_E = beta.pdf(p, alpha_post_neg_E, beta_post_neg_E)

# Test: H0 = π_E > π_¬E, H1 = π_E <= π_¬E
n_samples = 100000
samples_E = beta.rvs(alpha_post_E, beta_post_E, size=n_samples)
samples_neg_E = beta.rvs(alpha_post_neg_E, beta_post_neg_E, size=n_samples)

# Wahrscheinlichkeiten für H0 und H1 berechnen
p_H1 = np.mean(samples_E < samples_neg_E)
p_H0 = 1 - p_H1

# Ergebnisse ausgeben
print(f"Wahrscheinlichkeit für H0 (π_E >= π_¬E): {p_H0:.4f}")
print(f"Wahrscheinlichkeit für H1 (π_E < π_¬E): {p_H1:.4f}")
if p_H0 > p_H1:
    print("Die Nullhypothese H0 wird akzeptiert: π_E >= π_¬E")
else:
    print("Die Nullhypothese H0 wird abgelehnt: π_E < π_¬E")


# Plot der Verteilungen
plt.plot(p, prior_E, label=r'Priori (für europäische Teams)', color='blue')
plt.plot(p, prior_neg_E, label=r'Priori (für nicht-europäische Teams)', color='green')
plt.plot(p, likelihood_G, label=r'Likelihood (für alle Spiele)', color='red')
plt.plot(p, posterior_E, label=r'Posterior (für europäische Teams)', color='blue', linestyle='--')
plt.plot(p, posterior_neg_E, label=r'Posterior (für nicht-europäische Teams)', color='green', linestyle='--')
plt.title('Prior-, Likelihood- und Posterior-Verteilungen der Siegquote (Bayesianisch)')
plt.xlabel('Siegquote (p)')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True)

# Zeige alle Plots
plt.tight_layout()
plt.show()
