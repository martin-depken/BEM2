import numpy as np
from SimulatedAnnealing import sim_anneal_fit as simann
import matplotlib.pyplot as plt









# Create data
# x = np.linspace(0,100,1000)
A = 1.0
C = 0.75
nseed = 8
yerr = 0.05
x = np.arange(1,20)
y = A / (1 + np.exp(-(x-nseed) * C)) + yerr * np.random.rand(len(x))

# Fit parameters
start_params = np.array([5,35,6])
lwr_bnd = [0.0, 1.0, 0.0]
uppr_bnd  = [10.0,20.0, 4.0 ]


# Do fit
fit = simann(x,y,yerr,start_params,lwr_bnd,uppr_bnd,Tfinal=1.0)

# Plot results
plt.errorbar(x,y,yerr=yerr,fmt='x')
plt.plot(x, fit[0] / (1 + np.exp(-(x-fit[1]) * fit[2])))
plt.show()


