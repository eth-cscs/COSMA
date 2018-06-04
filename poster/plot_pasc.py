import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

n_groups = 7

means_carma = (1021.45, 253.97, 846.34, 839.34, 609.52, 877.37, 1025.38)
means_scalapack_opt = (420.10, 114.28, 492.31, 386.42, 256.00, 352.06, 274.23)
means_scalapack_naive = (284.84, 41.03, 279.02, 264.60, 196.92, 67.97, 54.50)

fig, ax = plt.subplots()

#index = np.arange(n_groups)
index = np.zeros(n_groups)
bar_width = 0.8

opacity = 1
gap = 0.2
midgap = 0.6

offset = 2

for i in range(n_groups):
    space = 2 * gap + 3* bar_width + midgap
    if (i == 0):
        index[i] = space
    else:
        index[i] = index[i - 1] + space
    if i == 2:
        index[i] = index[i] + offset
    if i == 4:
        index[i] = index[i] + offset

ax.bar(index, means_carma, bar_width,
    alpha=opacity, color="deepskyblue", 
    label='CARMA')

ax.bar(index + bar_width + gap, means_scalapack_naive, bar_width,
    alpha=opacity, color="gold",
    label='ScaLAPACK (naive)')

ax.bar(index + bar_width + bar_width + gap + gap, means_scalapack_opt, bar_width,
    alpha=opacity, color="orangered",
    label='ScaLAPACK (tuned) *')

ax.axhline(y=1209, linewidth=1, color = "green")
ax.annotate('peak throughput per node', xy=(2, 1), xytext=(3, 1240), color="green")

ax.set_ylabel('Throughput per node (GFlops/s)', fontsize=14)
ax.set_title('CARMA vs ScaLAPACK on 64 nodes on Piz Daint', fontsize=14)

ax.set_ylim([0, 1300])

#ax.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

ticks = [0, 0, 0]
index_idx = [0, 2, 4]
for i in range(3):
    next_idx = index_idx[i] + 1
    if i == 2:
        next_idx = next_idx + 1
    ticks[i] = index[index_idx[i]] + index[next_idx]  + 2 * bar_width + 2 * gap
    ticks[i] = ticks[i] / 2

ax.set_xticks(ticks)
ax.set_xticklabels(('square', 'two large\n dimensions', 'one large\n dimension'), fontsize=14)
ax.legend(loc='upper center',  bbox_to_anchor=(0.5, 0.92))

fig.tight_layout()
fig.savefig("results.pdf")
