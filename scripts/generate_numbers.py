import numpy as np
import math

#n_nodes = [4, 7, 8, 13, 16, 25, 27, 32, 37, 61, 64, 81, 93, 128, 201, 216, 256, 333, 473, 512]
n_nodes = [int(math.ceil(2**i/36.0)) for i in range(7, 15)]
n_tasks = [2**i for i in range(7, 15)]
p_range = [int(math.floor(2**i/36.0))*4 for i in range(7, 15)]
p_rows = [3, 4, 7, 8, 14, 4, 4, 35]
p_cols = [4, 7, 8, 14, 16, 113, 227, 52]

base_memory = 1250000000 # per node, in #doubles, corresponding to 10GB
mem_limit = 1.0 * base_memory

#p_range=[16, 28, 32, 52, 64, 100, 108, 128, 148, 244, 256, 324, 372, 512, 804, 864, 1024, 1332, 1892, 2048]
#p_rows=[4, 4, 4, 4, 8, 10, 9, 8, 4, 4, 16, 18, 12, 16, 12, 24, 32, 36, 43, 32]
#p_cols=[4, 7, 8, 13, 8, 10, 12, 16, 37, 61, 16, 18, 31, 32, 67, 36, 32, 37, 44, 64]

# these can be any values independent of available memory or nodes
strong_scaling_square=16384

strong_scaling_thin_mn=17408
strong_scaling_thin_k=3735552

def cubic_root(x):
    return x**(1.0/3.0)

def square_root(x):
    return math.sqrt(x)

def get_weak_scaling_p0_mn():
    return [int(math.floor(136.0*cubic_root(p*mem_limit/(3.0 * 228.0 * 136.0)))) for p in n_nodes]

def get_weak_scaling_p0_k():
    return [int(math.floor(228.0/(136.0**2.0) * (p**2.0))) for p in weak_scaling_p0_mn]

def get_weak_scaling_p1_mn():
    return [int(math.floor(136.0*cubic_root((p**(2.0/3.0))*mem_limit/(3.0 * 228.0 * 136.0)))) for p in n_nodes]

def get_weak_scaling_p1_k():
    return [int(math.floor(228.0/(136.0**2.0) * (p**2))) for p in weak_scaling_p1_mn]

def get_weak_scaling_p0():
    return [int(math.floor(square_root(p * mem_limit / 3.0))) for p in n_nodes]

def get_weak_scaling_p1():
    return [int(math.floor(square_root((p**(2.0/3.0)) * mem_limit / 3.0))) for p in n_nodes]

weak_scaling_p0 = get_weak_scaling_p0()
weak_scaling_p1 = get_weak_scaling_p1()

weak_scaling_p0_mn = get_weak_scaling_p0_mn()
weak_scaling_p0_k = get_weak_scaling_p0_k()

weak_scaling_p1_mn = get_weak_scaling_p1_mn()
weak_scaling_p1_k = get_weak_scaling_p1_k()

mem_limit = 4 * base_memory

def apply_correctness_factor(l, factor):
    #poly = np.polyfit([4, n_nodes[len(n_nodes)-1]], [factor, 0.1], deg=1)
    #factors = [np.polyval(poly, x) for x in n_nodes]
    #return [int(factors[i] * l[i]) for i in range(len(n_nodes))]
    return [int(factor * l[i]) for i in range(len(n_nodes))]

# 0.5 0.4 0.6 0.6 0.5 0.5
weak_scaling_p0 = apply_correctness_factor(weak_scaling_p0, 0.4)
weak_scaling_p1 = apply_correctness_factor(weak_scaling_p1, 0.4)

weak_scaling_p0_mn = apply_correctness_factor(weak_scaling_p0_mn, 0.4)
weak_scaling_p0_k = apply_correctness_factor(weak_scaling_p0_k, 0.4)

weak_scaling_p1_mn = apply_correctness_factor(weak_scaling_p1_mn, 0.4)
weak_scaling_p1_k = apply_correctness_factor(weak_scaling_p1_k, 0.4)

print(weak_scaling_p0)
print(weak_scaling_p1)

print(weak_scaling_p0_mn)
print(weak_scaling_p0_k)

print(weak_scaling_p1_mn)
print(weak_scaling_p1_k)

template_file = "generate_scripts.sh"
output_file = "generate_scripts_filled.sh"

def get_string(num_list):
    result = '('
    for x in num_list:
        result += str(x) + " "
    result += ')'
    return result

with open(template_file) as in_file:
    with open(output_file, 'w') as out_file:
        for line in in_file:
            line = line.replace("n_nodes=()", "n_nodes=" + get_string(n_nodes))
            line = line.replace("n_tasks=()", "n_tasks=" + get_string(n_tasks))

            line = line.replace("p_range=()", "p_range=" + get_string(p_range))
            line = line.replace("p_rows=()", "p_rows=" + get_string(p_rows))
            line = line.replace("p_cols=()", "p_cols=" + get_string(p_cols))

            line = line.replace("weak_scaling_p0=()", "weak_scaling_p0=" + get_string(weak_scaling_p0))
            line = line.replace("weak_scaling_p1=()", "weak_scaling_p1=" + get_string(weak_scaling_p1))

            line = line.replace("weak_scaling_p0_mn=()", "weak_scaling_p0_mn=" + get_string(weak_scaling_p0_mn))
            line = line.replace("weak_scaling_p0_k=()", "weak_scaling_p0_k=" + get_string(weak_scaling_p0_k))

            line = line.replace("weak_scaling_p1_mn=()", "weak_scaling_p1_mn=" + get_string(weak_scaling_p1_mn))
            line = line.replace("weak_scaling_p1_k=()", "weak_scaling_p1_k=" + get_string(weak_scaling_p1_k))

            line = line.replace("strong_scaling_square=()", "strong_scaling_square=" + str(strong_scaling_square))
            line = line.replace("strong_scaling_thin_mn=()", "strong_scaling_thin_mn=" + str(strong_scaling_thin_mn))
            line = line.replace("strong_scaling_thin_k=()", "strong_scaling_thin_k=" + str(strong_scaling_thin_k))

            line = line.replace("mem_limit=()", "mem_limit=" + str(mem_limit))

            out_file.write(line)
