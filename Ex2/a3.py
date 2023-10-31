import math

import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt

r = rand.rand()
print(r)


# used to sample with a bias
def dir_choice(bias = 0.5):
    if rand.rand() < bias:
        return -1
    else:
        return 1


# returns an array of dim d, N that shows all steps of a rw
def rand_walk(N: int, d: int, bias=0.5):
    rw = np.zeros((d, N))
    for i in range(1, N):
        add_at = rand.randint(0, d)
        rw[:, i] = rw[:, i-1]
        rw[add_at, i] += dir_choice(bias=bias)

    return rw


# generate N//stepsize rwalks n_runs times and return the average distances as vec of dim 2, N//stepsize
def generate_end_to_end_dist(N, d, stepsize=10, bias=0.5, n_runs=1):
    num_steps = N // stepsize
    avg_end_to_end_distances = np.zeros((2, num_steps))
    avg_end_to_end_distances[0] = [x for x in range(0, N, stepsize)]

    for _ in range(n_runs):
        end_to_end_distances = np.zeros((2, num_steps))
        for i in range(stepsize, N + 1, stepsize):
            rw = rand_walk(i, d, bias=bias)
            end_to_end_distances[:, i // stepsize - 1] = i, np.linalg.norm(rw[:, -1])
        avg_end_to_end_distances[1] += end_to_end_distances[1]

    # average the summed end_to_end_distances
    avg_end_to_end_distances[1] = np.divide(avg_end_to_end_distances[1], n_runs)
    return avg_end_to_end_distances


distances = generate_end_to_end_dist(1000, 2, n_runs=100)
print(distances)


fig1, ax1 = plt.subplots()
ax1.plot(distances[0], distances[1])
ax1.set_xlabel('Number of Steps')
ax1.set_ylabel('Average Distance')
ax1.set_title('Average Distance vs. Number of Steps averaged over 100 runs')
fig1.savefig('a3_1.pdf')
# plt.show()


#######
# 3.3 #
#######


fig2, ax2 = plt.subplots()
distances = generate_end_to_end_dist(1000, 1, n_runs=10, bias=0.3)
ax2.plot(distances[0], distances[1], label='bias = 0.7')
distances = generate_end_to_end_dist(1000, 1, n_runs=10, bias=0.2)
ax2.plot(distances[0], distances[1], label='bias = 0.8')
distances = generate_end_to_end_dist(1000, 1, n_runs=10, bias=0.1)
ax2.plot(distances[0], distances[1], label='bias = 0.9')
distances = generate_end_to_end_dist(1000, 1, n_runs=10, bias=0.0)
ax2.plot(distances[0], distances[1], label='bias = 1')
plt.legend()


ax2.set_xlabel('Number of Steps')
ax2.set_ylabel('Average Distance')
ax2.set_title('Average Distance vs. Number of Steps averaged over 100 runs with bias')
fig2.savefig('a3_3.pdf')
plt.show()


####### What changes and why?
# Due to the bias the average distance increases, since the steps random walk are less likely to reverse previouse steps
