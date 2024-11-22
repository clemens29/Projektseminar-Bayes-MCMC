import numpy as np
import matplotlib.pyplot as plt

# Wahrer Mittelwert und Standardabweichung
true_mean = 10
true_sd = 2

# Stichprobe von 10 Werten ziehen
np.random.seed(40)
sample_size = 10
data = np.random.normal(loc=true_mean, scale=true_sd, size=sample_size)
print(data)

# Bayes'sche Schätzung: Mittelwert der Stichprobe

def likelihood(param, data):
    mu, sigma = param
    # Likelihood-Funktion für Normalverteilung
    return -0.5 * np.sum(np.log(2 * np.pi * sigma**2) + ((data - mu)**2) / (sigma**2))

def prior(param):
    mu, sigma = param
    # Prior: Normalverteilung für mu und Uniformverteilung für sigma
    prior_mu = -0.5 * (mu**2) / 20**2  # Normal prior mit mu = 0 und sigma = 10
    prior_sigma = -np.log(sigma) if sigma > 0 else -np.inf  # Uniform prior für sigma
    return prior_mu + prior_sigma

def posterior(param, data):
    return likelihood(param, data) + prior(param)

def metropolis_hastings(data, start_value, iterations):
    # Startwert für [mu, sigma]
    chain = np.zeros((iterations, 2))
    chain[0, :] = start_value
    
    for i in range(1, iterations):
        # Vorschlagswert basierend auf Normalverteilung
        proposal = np.random.normal(loc=chain[i-1, :], scale=[0.5, 0.5])
        
        # Akzeptanzwahrscheinlichkeit
        prob_accept = min(1, np.exp(posterior(proposal, data) - posterior(chain[i-1, :], data)))
        
        if np.random.rand() < prob_accept:
            chain[i, :] = proposal
        else:
            chain[i, :] = chain[i-1, :]
    
    return chain

# MCMC-Startwert und Iterationen
start_value = [np.mean(data), np.std(data)]  # Startwerte für [mu, sigma]
iterations = 10000
chain = metropolis_hastings(data, start_value, iterations)

# Burn-In Phase entfernen
burn_in = 5000
chain_burned = chain[burn_in:]

# Posterior-Analyse
mu_samples = chain_burned[:, 0]
sigma_samples = chain_burned[:, 1]


# Schätzung der Parameter aus der Posterior-Verteilung
mu_estimate = np.mean(mu_samples)
sigma_estimate = np.mean(sigma_samples)

print(f"Bayes'sche Schätzung des Mittelwerts (mu): {mu_estimate}")
print(f"Bayes'sche Schätzung der Standardabweichung (sigma): {sigma_estimate}")

# Konfidenzintervall berechnen
ci_mu = np.percentile(mu_samples, [2.5, 97.5])
ci_sigma = np.percentile(sigma_samples, [2.5, 97.5])

print(f"95% Konfidenzintervall für mu: {ci_mu}")
print(f"95% Konfidenzintervall für sigma: {ci_sigma}")

# Plot der Samples
plt.figure(figsize=(12, 6))

# Posterior des Mittelwerts (mu)
plt.subplot(1, 2, 1)
plt.hist(mu_samples, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(true_mean, color='red', linestyle='--', label=f'True Mean = {true_mean}')
plt.title("Posterior of mu")
plt.xlabel("mu")
plt.ylabel("Frequency")
plt.legend()

# Posterior der Standardabweichung (sigma)
plt.subplot(1, 2, 2)
plt.hist(sigma_samples, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(true_sd, color='red', linestyle='--', label=f'True SD = {true_sd}')
plt.title("Posterior of sigma")
plt.xlabel("sigma")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()