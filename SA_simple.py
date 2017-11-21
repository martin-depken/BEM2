#################################
# Basic Simmulated Annealing
# (to fit an exponential to data using least squares)
#
# Misha Klein
#
#
##################################
import numpy as np


'''
Main function
'''
def sim_anneal_fit(xdata, ydata, yerr, p_start, Tstart, Tfinal, delta0, alpha=0.85):
    '''
    :param xdata: datapoints (indep. variable)
    :param ydata: datapoints (values)
    :param yerr: measurement error
    :param p_start: starting guess for parameters
    :param Tstart: Initial Temperature
    :param Tfinal: Final Temperature. Sets stop condition
    :param delta0: Stepsize
    :param alpha: Cooling rate. Exponential cooling is used
    :return: Optimal parameter set (p)
    '''
    T = Tstart
    delta = delta0  # only needed if you actively adjust stepsize
    p = p_start
    step = 0

    while (T > Tfinal):
        step += 1
        if (step % 100 == 0):
            #             print T
            T = update_temp(T, alpha)
        p_trial = p + np.random.uniform(-delta, delta, size=len(p))
        Vnew = V(xdata, ydata, yerr, p_trial)
        Vold = V(xdata, ydata, yerr, p)

        if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
            p = p_trial
    return p



'''
Functions called from main function
'''

def update_temp(T, alpha):
    '''
    Exponential cooling scheme
    :param T: current temperature.
    :param alpha: Cooling rate.
    :return: new temperature
    '''
    T *= alpha
    return T


'''
Model: Exponential
'''
# Residual/Penalty function: serves as the "potential"
def V(xdata, ydata, yerr, params):
    k = params[0]
    model = f(xdata, k)
    return np.sum(((model - ydata) / yerr) ** 2)


def f(x, k):
    return np.exp(-k * x)