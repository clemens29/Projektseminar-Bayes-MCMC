import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# Wahre Parameter und Daten
trueA = 5
trueB = 0
trueSd = 10
sampleSize = 31

# Erstellen von x- und y-Daten
x = np.arange(-(sampleSize - 1) / 2, (sampleSize - 1) / 2 + 1)
y = trueA * x + trueB + np.random.normal(0, trueSd, size=sampleSize)

# Likelihood-Funktion
def likelihood(param):
    a, b, sd = param
    pred = a * x + b
    single_likelihoods = norm.logpdf(y, loc=pred, scale=sd)
    return np.sum(single_likelihoods)

# Prior-Funktion
def prior(param):
    a, b, sd = param
    
    # Uniform Prior für a und b
    prior_a = uniform.logpdf(a, loc=-10, scale=20)  # Uniform von -10 bis 10
    prior_b = uniform.logpdf(b, loc=-20, scale=40)  # Uniform von -20 bis 20
    
    # Normal Prior für sd (positiv begrenzt)
    prior_sd = norm.logpdf(sd, loc=0, scale=10) if sd > 0 else -np.inf
    
    return prior_a + prior_b + prior_sd

# Posterior-Funktion
def posterior(param):
    return likelihood(param) + prior(param)

# Vorschlagsfunktion
def proposal_function(param):
    # Normalverteilung für Vorschläge mit unterschiedlicher Streuung
    return np.random.normal(loc=param, scale=[0.1, 0.5, 0.3])

# Metropolis-Hastings-Algorithmus
def run_metropolis_mcmc(start_value, iterations):
    chain = np.zeros((iterations + 1, len(start_value)))
    chain[0, :] = start_value
    
    for i in range(iterations):
        proposal = proposal_function(chain[i, :])
        
        # Akzeptanzwahrscheinlichkeit
        probab = np.exp(posterior(proposal) - posterior(chain[i, :]))
        if np.random.rand() < probab:
            chain[i + 1, :] = proposal
        else:
            chain[i + 1, :] = chain[i, :]
    
    return chain

# Parameter und Iterationen
start_value = [4, 0, 10]  # Startwerte für [a, b, sd]
iterations = 10000
chain = run_metropolis_mcmc(start_value, iterations)

# Burn-In-Phase entfernen
burn_in = 5000
chain_burned = chain[burn_in:]

# Akzeptanzrate berechnen
acceptance_rate = 1 - np.mean(np.all(chain_burned[:-1] == chain_burned[1:], axis=1))
print(f"Akzeptanzrate: {acceptance_rate:.2f}")

import statsmodels.api as sm

# Lineare Regression mit Statsmodels
x_with_const = sm.add_constant(x)  # Addiere eine Konstante für den Intercept
model = sm.OLS(y, x_with_const)    # Erstelle ein OLS-Modell
results = model.fit()              # Fitte das Modell

# Ausgabe der Regressionsergebnisse
print(results.summary())

# Vergleich mit den MCMC-Ergebnissen
print("\nVergleich mit MCMC-Ergebnissen:")
print(f"Wahrer Wert von a (Steigung): {trueA}")
print(f"Schätzung von a (Posterior Mean): {np.mean(chain_burned[:, 0]):.4f}")
print(f"Wahrer Wert von b (Intercept): {trueB}")
print(f"Schätzung von b (Posterior Mean): {np.mean(chain_burned[:, 1]):.4f}")
print(f"Wahrer Wert von sd: {trueSd}")
print(f"Schätzung von sd (Posterior Mean): {np.mean(chain_burned[:, 2]):.4f}")

# Ergebnisse plotten
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histogramme für die Posterior-Verteilungen
axes[0, 0].hist(chain_burned[:, 0], bins=30, color='skyblue', alpha=0.7)
axes[0, 0].axvline(np.mean(chain_burned[:, 0]), color='blue', linestyle='--', label='Mean')
axes[0, 0].axvline(trueA, color='red', linestyle='-', label='True value')
axes[0, 0].set_title("Posterior of a")
axes[0, 0].set_xlabel("a")
axes[0, 0].legend()

axes[0, 1].hist(chain_burned[:, 1], bins=30, color='skyblue', alpha=0.7)
axes[0, 1].axvline(np.mean(chain_burned[:, 1]), color='blue', linestyle='--', label='Mean')
axes[0, 1].axvline(trueB, color='red', linestyle='-', label='True value')
axes[0, 1].set_title("Posterior of b")
axes[0, 1].set_xlabel("b")
axes[0, 1].legend()

axes[0, 2].hist(chain_burned[:, 2], bins=30, color='skyblue', alpha=0.7)
axes[0, 2].axvline(np.mean(chain_burned[:, 2]), color='blue', linestyle='--', label='Mean')
axes[0, 2].axvline(trueSd, color='red', linestyle='-', label='True value')
axes[0, 2].set_title("Posterior of sd")
axes[0, 2].set_xlabel("sd")
axes[0, 2].legend()

# Verlauf der Ketten
axes[1, 0].plot(chain_burned[:, 0], color='blue')
axes[1, 0].axhline(trueA, color='red', linestyle='--', label='True value')
axes[1, 0].set_title("Chain values of a")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].legend()

axes[1, 1].plot(chain_burned[:, 1], color='blue')
axes[1, 1].axhline(trueB, color='red', linestyle='--', label='True value')
axes[1, 1].set_title("Chain values of b")
axes[1, 1].set_xlabel("Iteration")
axes[1, 1].legend()

axes[1, 2].plot(chain_burned[:, 2], color='blue')
axes[1, 2].axhline(trueSd, color='red', linestyle='--', label='True value')
axes[1, 2].set_title("Chain values of sd")
axes[1, 2].set_xlabel("Iteration")
axes[1, 2].legend()

plt.tight_layout()
plt.show()