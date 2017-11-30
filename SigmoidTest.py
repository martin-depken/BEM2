import numpy as np
from SimulatedAnnealing import sim_anneal_fit as simann
import matplotlib.pyplot as plt


# create data
A = 1.0
C = 0.75
nseed = 8
x = np.arange(1,21)
yerr = [0.05]*len(x)
y = A / (1 + np.exp(-(x-nseed) * C)) + yerr * np.random.rand(len(x))

# Fit parameters
start_params = np.array([0.6,35,6])
lwr_bnd = [0.0, 1.0, 0.0]
uppr_bnd  = [15.0,20.0, 4.0 ]


# Do fit
fit = simann(x,y,yerr,start_params,lwr_bnd,uppr_bnd,use_multiprocessing=True,nprocs=4,Tfinal=1.0,tol=0.001,Tstart=500000)
print fit

# Plot results
plt.plot(x, fit[0] / (1 + np.exp(-(x-fit[1]) * fit[2])))
plt.errorbar(x,y,yerr=yerr,fmt='x')
plt.show()


