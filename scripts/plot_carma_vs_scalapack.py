import numpy as np
import matplotlib.pyplot as plt

carma_file="carma.txt"
scalapack_file="scalapack.txt"

nodes = [16, 32, 48, 64]
m = 8704
n = m
k = 933888
length = len(nodes)

tps = m * n * k * 2.0 / (1e9)

f = open(carma_file, 'r')
carma_time = f.read().splitlines()
f.close()
carma_time = [float(num)/1000.0 for num in carma_time]
carma_tps = [tps / nodes[i] / carma_time[i] for i in range(length)]

f = open(scalapack_file, 'r')
scalapack_time = f.read().splitlines()
f.close()
scalapack_time = [float(num)/1000.0 for num in scalapack_time]
scalapack_tps = [tps / nodes[i] / scalapack_time[i] for i in range(length)]

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(2, 2, 1)
ax.set_title("Comparison of times")
ax.set_xlabel('Nodes (36 ranks / node)')
ax.set_ylabel('Time [s]')
ax.plot(nodes, carma_time, label="CARMA")
ax.plot(nodes, scalapack_time, label="SCALAPACK")
ax.legend()

ax = fig.add_subplot(2, 2, 2)
ax.set_title("Relative Time Speedup")
ax.set_xlabel('Nodes (36 ranks / node)')
ax.set_ylabel('Time(Scalapack)/Time(CARMA)')
ax.plot(nodes, [scalapack_time[i] / carma_time[i] for i in range(length)])

ax = fig.add_subplot(2, 2, 3)
ax.set_title("Comparison of TPS (GFlops/s)")
ax.set_xlabel('Nodes (36 ranks / node)')
ax.set_ylabel('GFlops/s')
ax.plot(nodes, carma_tps, label="CARMA")
ax.plot(nodes, scalapack_tps, label="SCALAPACK")
ax.legend()

ax = fig.add_subplot(2, 2, 4)
ax.set_title("Relative TPS Speedup")
ax.set_xlabel('Nodes (36 ranks / node)')
ax.set_ylabel('TPS(CARMA)/TPS(Scalapack)')
ax.plot(nodes, [carma_tps[i] / scalapack_tps[i] for i in range(length)])

fig.tight_layout()

fig.savefig("carma_vs_scalapack.pdf")
