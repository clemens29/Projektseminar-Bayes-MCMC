import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

m = 0.03
v = 0.021
f_v_m = 0.5

ln_f_v_m = np.log(f_v_m)
ln_ratio_v_m = np.log(v / m)
ln_ratio_1_v_1_m = np.log((1 - v) / (1 - m))
factor = (1 - m) / m

x = 1989
n_x = 20042
y = 1883
n_y = 56104

# Priorparameter
a_prior = ln_f_v_m / (ln_ratio_v_m + factor * ln_ratio_1_v_1_m) + 1
b_prior = (a_prior-1) * factor + 1

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

#Calculate the 99% credible intervals
ci_X = beta.interval(0.99, a_X, b_X)
ci_Y = beta.interval(0.99, a_Y, b_Y)

ci_X = (round(ci_X[0], 4), round(ci_X[1], 4))
ci_Y = (round(ci_Y[0], 4), round(ci_Y[1], 4))

print('99% Kredibilitätsintervall für X:', ci_X)
print('99% Kredibilitätsintervall für Y:', ci_Y)

# Plot the prior and posterior distributions
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