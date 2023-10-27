import numpy as np
import matplotlib.pyplot as plt

#######
# 1.a #
#######
unif = np.random.uniform(-10, -4, 10000)

#######
# 1.b #
#######

norm = np.random.randn(10000)

#######
# 1.b #
#######

pois = np.random.poisson(3, 10000)

fig, axs = plt.subplots(2,2)
fig.suptitle('1 b')
axs[0, 0].hist(unif)
axs[0, 0].set_xlabel('Value')
axs[0, 0].set_ylabel('Count')
axs[0, 1].hist(norm)
axs[0, 1].set_xlabel('Value')
axs[0, 1].set_ylabel('Count')
axs[1, 0].hist(pois)
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Count')

fig.tight_layout()
fig.savefig('dist_overview.pdf')

for ax in axs.flat:
    ax.set(xlabel='Value', ylabel='Counts')
    # ax.legend()

plt.tight_layout()
# plt.show()


