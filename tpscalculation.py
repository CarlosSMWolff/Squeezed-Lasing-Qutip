from qutip import *
import numpy as np
from numpy import sqrt
import time
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from TPS_simulation import *
from plot_functions import *

qutip.settings.num_cpus = 24

g = 1
ratio = 0.01
C = 1.
P = sqrt(4/(C*ratio))
kappa = ratio*P
gsens = 0.0001
Gamma = 0.01

# Squeezing parameter
r = 0.4

# Truncation parameter
Ncav = 30

sim = simulation()

nw = 3
wgrid = np.linspace(-.1,0.1,nw)

# 1D list
DeltaList=list(zip(wgrid,-wgrid))

# 2D list
DeltaList2D = []
for Delta1 in wgrid:
    for Delta2 in wgrid:
        DeltaList2D.append([Delta1,Delta2])

r=0.4

tic = time.time()
g2w1w2Parallel = np.array(parallel_map(sim.g2g1,DeltaList2D,(g,gsens,kappa,P,Gamma,r),progress_bar=True))
toc = time.time()

print(f'Time was {toc-tic} seconds')

g2w1w22D = g2w1w2Parallel.reshape((nw,nw))
np.save(f'g2_w1w2_2D-r-{r}-nw-{nw}',g2w1w22D)