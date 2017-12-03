import numpy as np
import matplotlib.pylab as plt
import SimulatedAnnealing as SA
from scipy.linalg import expm



'''
example models


make your own model to fit to data in the form:
def my_model(datapoints, parameters):
    .
    .
    .
    return
'''
def Exponential(t, params):
    # Naturally, this seems redundant, but just for the sake of illustration:
    rate = params[0]
    return rate*np.exp(-rate*t)

def Sigmoid(x,params):
    amplitude = params[0]
    half_saturation = params[1]
    decay = params[2]
    return amplitude/(1+ np.exp(-(x-half_saturation)*decay))

def Line(x,params):
    slope = params[0]
    offset = params[1]
    return slope*x + offset
    
def MatrixExponential(x,params):
    factor = params[0]
    matrix_size = 20
    matrix = np.zeros((matrix_size,matrix_size))
    
    # Build a tridiagonal matrix
    for d in range(-1,2):
        diagonal = np.linspace(-10,10,num=(matrix_size-abs(d)))
        matrix = matrix + np.diag(diagonal,k=d)
        
    # Exponentiate the matrix
    y = []
    for x_point in x:
        matrix_exp = factor*expm(matrix*x_point)
        y.append(matrix_exp[10,10])
    
    return y
    
    

'''
select model

adjust this part depending on what model you want to fit
'''
my_model = MatrixExponential  # select the name of the model function
amplitude = 0.7
half_saturation = 11
decay = 2.5
parameter_values = [amplitude]


'''
Generating 'mock data'
'''
datapoints = np.linspace(1,21,100)
error_amp = [0.05] * len(datapoints)
experimental_values = my_model(datapoints,parameter_values) + error_amp * np.random.rand(len(datapoints))

'''
presets for the fit

(adjust the number of values in the initial guess and upper and lower bounds to suite the model chosen)
'''
starting_guess = [1.0]
lwr_bnd = [0.0]    # using np.inf to not use this bound (set it to +/- infinity)
uppr_bnd  = [np.inf]


'''
perform fit // optimisation using simulated annealing

here we also set some of the optional parameters
'''
fit_result = SA.sim_anneal_fit(model=my_model,
             xdata=datapoints,
             ydata=experimental_values,
             yerr=error_amp,
             Xstart= starting_guess,
             lwrbnd=lwr_bnd,
             upbnd=uppr_bnd,
             use_multiprocessing=True,
             nprocs=4,
             Tfinal=0.5,
              tol=0.001,
             Tstart=1000)


print "input parameter set: ", parameter_values
print "fitted parameter set: ", fit_result


'''
plot the result
'''

# Plot results
plt.title('Fitting result')
plt.plot(datapoints,my_model(datapoints,starting_guess),label='starting guess',linestyle='dashed')
plt.plot(datapoints, my_model(datapoints,fit_result),label='fit')
plt.errorbar(datapoints,experimental_values,yerr=error_amp,fmt='x', label='data')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y',rotation=0)
# plt.show()

'''
double check: using the output file

(this you must do in case of manual interupt)
'''
# fatch the last recorded parameter set:
X = np.loadtxt('fit_results.txt')
fit_result = X[-1,:]

# Plot results
plt.figure()
plt.title('Retreiving the final stored parameter set')
plt.plot(datapoints,my_model(datapoints,starting_guess),label='starting guess',linestyle='dashed')
plt.plot(datapoints, my_model(datapoints,fit_result),label='fit')
plt.errorbar(datapoints,experimental_values,yerr=error_amp,fmt='x', label='data')
plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.show()
