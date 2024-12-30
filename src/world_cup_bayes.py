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
alpha_post_E = alpha_E + 36
beta_post_E = beta_E + 29
alpha_post_neg_E = alpha_neg_E + 36
beta_post_neg_E = beta_neg_E + 11

# Erwartungswert der Posterior-Verteilungen
E_post_E = alpha_post_E / (alpha_post_E + beta_post_E)
E_post_neg_E = alpha_post_neg_E / (alpha_post_neg_E + beta_post_neg_E)

# Berechnen des 95%-Konfidenzintervalls
CI_post_E = beta.ppf([0.025, 0.975], alpha_post_E, beta_post_E)
CI_post_neg_E = beta.ppf([0.025, 0.975], alpha_post_neg_E, beta_post_neg_E)

# Ausgabe der Erwartungswerte und Konfidenzintervalle
print(f"Erwartungswert für europäische Teams (π_E): {E_post_E:.4f}")
print(f"Erwartungswert für nicht-europäische Teams (π_¬E): {E_post_neg_E:.4f}")
print(f"95%-Konfidenzintervall für π_E: [{CI_post_E[0]:.4f}, {CI_post_E[1]:.4f}]")
print(f"95%-Konfidenzintervall für π_¬E: [{CI_post_neg_E[0]:.4f}, {CI_post_neg_E[1]:.4f}]")

# --- Bayesianischer Hypothesentest ---
# Erzeugen von Zufallszahlen aus den Posterior-Verteilungen
post_E_samples = beta.rvs(alpha_post_E, beta_post_E, size=10000)
post_neg_E_samples = beta.rvs(alpha_post_neg_E, beta_post_neg_E, size=10000)

# Berechnung des Unterschieds der Posterior-Verteilungen
D_samples = post_E_samples - post_neg_E_samples

# Berechnung der Differenz der Erwartungswerte
D_mean = np.mean(D_samples)

# Berechnung des 95%-Konfidenzintervalls für den Unterschied
CI_D = np.percentile(D_samples, [2.5, 97.5])

# Ausgabe des Testergebnisses
print(f"Erwartungswert des Unterschieds D = π_E - π_¬E: {D_mean:.4f}")
print(f"95%-Konfidenzintervall für den Unterschied D: [{CI_D[0]:.4f}, {CI_D[1]:.4f}]")

# Hypothesentest: Nullhypothese H0: D = 0
if CI_D[0] > 0 or CI_D[1] < 0:
    print("Die Nullhypothese (D = 0) wird abgelehnt. Es gibt einen signifikanten Unterschied.")
else:
    print("Die Nullhypothese (D = 0) kann nicht abgelehnt werden. Kein signifikanter Unterschied.")

# --- Plot 1: Bayesianischer Hypothesentest ---
plt.figure(figsize=(10, 6))

# Plot für Prior-, Likelihood- und Posterior-Verteilungen
p = np.linspace(0, 1, 1000)
prior_E = beta.pdf(p, alpha_E, beta_E)
prior_neg_E = beta.pdf(p, alpha_neg_E, beta_neg_E)
likelihood_G = beta.pdf(p, alpha_total, beta_total)

# Funktion für den Posterior
def posterior(prior, likelihood):
    return prior * likelihood

# Posterior für europäische und nicht-europäische Teams
posterior_E = posterior(prior_E, likelihood_G)
posterior_neg_E = posterior(prior_neg_E, likelihood_G)

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

# Plot für den Unterschied der Posterior-Verteilungen (D)
plt.figure(figsize=(10, 6))
plt.hist(D_samples, bins=50, density=True, alpha=0.6, color='g')
plt.axvline(x=0, color='r', linestyle='--', label='Nullhypothese: D = 0')
plt.title('Posterior-Verteilung des Unterschieds D = π_E - π_¬E')
plt.xlabel('Differenz der Siegquoten (D)')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True)

# Zeige alle Plots
plt.tight_layout()
plt.show()
