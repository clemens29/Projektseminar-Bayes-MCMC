import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_weeks = 100000  # Total number of weeks to simulate
positions = np.zeros(num_weeks, dtype=int)  # Track the island positions
current = 10  # Start at island 10 (the largest island)

# Simulation
for i in range(num_weeks):
    # Record the current position
    positions[i] = current
    
    # Flip coin to decide proposal (clockwise or counterclockwise)
    proposal = current + np.random.choice([-1, 1])
    
    # Wrap around if proposal goes out of bounds (circular archipelago)
    if proposal < 1:
        proposal = 10
    elif proposal > 10:
        proposal = 1
    
    # Compute the probability of moving to the proposal island
    prob_move = proposal / current  # Population ratio of proposal to current island
    
    # Move to proposal island if random number is less than prob_move
    if np.random.rand() < prob_move:
        current = proposal

# Visualization of results

# Plot king's position over the first 100 weeks
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(100), positions[:100], marker='o', linestyle='', markersize=4)
plt.xlabel('Week')
plt.ylabel('Island')
plt.title("King's Position Over First 100 Weeks")

# Plot distribution of time spent on each island
plt.subplot(1, 2, 2)
island_counts = np.bincount(positions[1:], minlength=11)[1:]  # Exclude island "0" placeholder
plt.bar(range(1, 11), island_counts)
plt.xlabel('Island')
plt.ylabel('Number of Weeks')
plt.title('Time Spent on Each Island')
plt.show()