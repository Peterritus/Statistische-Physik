import math

import numpy as np
import matplotlib.pyplot as plt


N_max = 1000

#######
# 2.a #
#######


def grid_rw(N):
    arr = np.zeros((2, N))
    for i in range(1, N):
        if np.random.rand() < 0.5:
            arr[0, i] = arr[0, i - 1] - 1
        else:
            arr[0, i] = arr[0, i - 1] + 1

        if np.random.rand() < 0.5:
            arr[1, i] = arr[1, i - 1] - 1
        else:
            arr[1, i] = arr[1, i - 1] + 1

    return arr

#######
# 2.b #
#######

fig, axs = plt.subplots(2, 2)

for i in range(2):
    for j in range(2):
        rw = grid_rw(N_max)
        axs[i, j].plot(rw[0], rw[1])
        axs[i, j].set_xlabel('X Coordinate')
        axs[i, j].set_ylabel('Y Coordinate')

fig.suptitle('Trajectory of Random Walk')
plt.tight_layout()
fig.savefig('Random Walk.pdf')
# plt.show()


#######
# 2.c #
#######

# Calculate average distance for different step counts
step_counts = np.arange(10, N_max + 1, 10)
avg_distances = []
avg_errors = []

for steps in step_counts:
    distances = []
    for _ in range(1000):
        rw = grid_rw(steps)
        final_distance = rw[0, -1] ** 2 + rw[1, -1] ** 2
        distances.append(final_distance)

    avg_distance = np.mean(distances)
    avg_distances.append(avg_distance)
    avg_error = np.std(distances)
    avg_errors.append(avg_error)

# linear fit
fit = np.polyfit(step_counts, avg_distances, 1)
best_fit_line = np.poly1d(fit)

# plotting
fig2 = plt.figure(2)

x_values = np.linspace(step_counts.min(), step_counts.max(), 100)
plt.plot(x_values, best_fit_line(x_values), label='Linear Fit')
plt.errorbar(step_counts, avg_distances, yerr=avg_errors, label='Values')

# Add labels and legend
plt.xlabel('Number of Steps')
plt.ylabel('Average Distance')
plt.title('Average Distance vs. Number of Steps')
plt.legend()
plt.grid(True)
fig2.savefig('Avg dist vs Step number.pdf')

#######
# 2.d #
#######

distances = []
step_counts = [1000, 2000, 3000]

for i in step_counts:
    for j in range(1000):
        rw = grid_rw(i)
        distances.append((i / 1000, rw[0, -1], rw[1, -1]))

# Create a figure with 2x2 subplots
fig3, axs = plt.subplots(2, 2)

# Create histograms for different step counts
for i, step_count in enumerate(step_counts):
    final_positions = [dist[1] for dist in distances if dist[0] == step_count / 1000]
    ax = axs[i // 2, i % 2]  # Calculate the subplot position
    ax.hist(final_positions, bins=20, edgecolor='black')
    ax.set_xlabel(f'Final X Position (Steps={step_count})')
    ax.set_ylabel('End to End Distance')

fig3.suptitle('E')
# Adjust the layout
plt.tight_layout()

# Show the subplots
plt.show()
fig3.savefig('hist ')
