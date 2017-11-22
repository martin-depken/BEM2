import numpy as np
from SimulatedAnnealing_MultiProcessing import sim_anneal_fit as simann
import matplotlib.pyplot as plt

# Create data
x = np.linspace(0,100,1000)
y = 10 / ( 1 + np.exp( (40-x) * 20 )  ) + np.random.rand(len(x))
yerr = np.array([1]*len(x))

# Fit parameters
start_params = np.array([3,30,13])
lwr_bnd = np.array([-100,-100,-100])
uppr_bnd = np.array([100,100,100])

# Do fit
fit = simann(x,y,yerr,start_params,lwr_bnd,uppr_bnd)

# Plot results
plt.errorbar(x,y,yerr=yerr,fmt='x')
plt.plot(x, fit[0] / (1 + np.exp(-(x-fit[1]) * fit[2])))
plt.show()


