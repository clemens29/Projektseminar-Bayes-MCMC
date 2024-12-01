import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

x = 1989
n_x = 20042
y = 1883
n_y = 56104

# Priorparameter
a_prior = 10
b_prior = 438

# Posteriorschätzer für Pi_X
a_X = a_prior + x
b_X = b_prior + n_x - x

# Posteriorschätzer für Pi_Y
a_Y = a_prior + y
b_Y = b_prior + n_y - y

xl = np.linspace(0, 0.12, 1000) 

# Calculate the prior and posterior distributions
prior = beta.pdf(xl, a_prior, b_prior)
posterior_X = beta.pdf(xl, a_X, b_X)
posterior_Y = beta.pdf(xl, a_Y, b_Y)

plt.figure(figsize=(10, 6))
plt.plot(xl, prior, label='Prioriverteilung', color='blue', linestyle='--', linewidth=2)
plt.plot(xl, posterior_X, label='Posterioriverteilung für X', color='green', linewidth=2)
plt.plot(xl, posterior_Y, label='Posterioriverteilung für Y', color='red', linewidth=2)

plt.title('Prior and Posterior Distributions', fontsize=16)
plt.xlabel('Sterblichkeitsrate', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()