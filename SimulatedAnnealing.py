##############################################################
#  Simulated Annealing for optimisation
#  Misha Klein
#
#
#
#
#
#############################################################
import numpy as np

'''
Main function
'''
def sim_anneal_fit(xdata, ydata, yerr, Xstart, lwrbnd, upbnd):
    '''
    Use Simmulated Annealing to perform Least-Square Fitting

    :param xdata: measured datapoints (independent variable)
    :param ydata: measured datapoints (value)
    :param yerr:  measurement error
    :param Xstart: first guess for parameter values
    :param lwrbnd: lower bound parameters
    :param upbnd:  upper bound parameters. Parameter X[i] will satisfy: lwrbnd <= X[i] <= upbnd[i]
    :return: Optimal parameter set (X, an array)
    '''

    # presets
    X = Xstart
    SA = SimAnneal()

    # Adjust initial temperature
    InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)

    # Set initial trial:
    X = Xstart
    SA.potential = V(xdata,ydata,yerr,X)
    
    steps = 0
    E = 0
    while True:
        steps += 1

        # I am starting to check If I switch temperature or if I stop simulation
        if SA.EQ:
            # E will represent the average energy at the current temperature
            # during the cycle of SA.interval steps that has just passed.
            E += V(xdata, ydata, yerr, X)

        if (steps % SA.interval == 0):
            if SA.EQ:
                E /= SA.interval
                Enew = E

                # Input this into the check for the stopcondition
                # Call update_temperature() and checks if global stop condition is reached:
                Temperature_Cycle(SA, xdata, ydata, yerr, X, Xstart)

                # Reset the cummalative sum:
                E = 0

                if SA.StopCondition:
                    break

            # updates stepsize based on Acceptance ratio and checks if you will update
            #  Temperature next time around (updates value "SimAnneal.EQ" to "TRUE"):
            AcceptanceRatio(SA)


        # Accept or reject trial configuration based on Metropolis Monte Carlo.
        # Input: parameters X, output: updates values of parameters X if accepted
        X = Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)

    print 'Final Temp: ', SA.T
    print 'Final Stepsize: ', SA.step_size
    return X



'''
Model (Potential)
'''
def sigmoid(A, nseed, C, x):
    return A / (1 + np.exp(-(x-nseed) * C))

def V(xdata,ydata,yerr,params):
    '''
    SHOULD ADJUST SUCH THAT THIS WORKS FOR ANY MODEL.
    :param xdata: datapoints
    :param ydata: measured values
    :param yerr: measurement error
    :param params: parameters of model to fit
    :return: Chi^2 value. Residual of weighted least squares (chi-squared).
    '''
    A = params[0]
    nseed = params[1]
    C = params[2]
    model = sigmoid(A,nseed,C,xdata)
    return np.sum(( (model-ydata)/yerr)**2    )


'''
Ancillerary functions
'''
class SimAnneal():
    '''
    stores all global parameters/settings of the Simmulated Annealing problem

    INPUT
    -----
    Tstart : Starting temperature (should not matter, is reset during inital loop)
    delta : initial step size for parameter changes
    tol : Stop condition. If relative change in values is less then tolerance, you're done.
    cooling_rate: Exponential cooling is used (T = T0^{cooling_rate})
    N_int : Number of steps between every check of the acceptance ratio / equillibrium
    AR_low: 'lower bound ideal acceptance ratio'
    AR_high: 'upper bound ideal acceptance ratio'. Stepsize (and initially temperature)
              are adjusted to keep the instantaneous acceptance ratio between AR_low and AR_high
    adjust_factor: Adjust the stepsize or temperature to have appreaciable acceptance ratio

    FURTHER MONITORS
    ----------------
    self.EQ: Will you move to next temperature? Did you equillibrate at current temperature?
    self.StopcCondition: Global criteria to stop the optimisation procedure.
    '''

    def __init__(self, Tstart=1.0, delta=2.0, tol=1E-1, adjust_factor=1.1, cooling_rate=0.85, N_int=1000,
                 AR_low=40, AR_high=60):
        self.T = Tstart
        self.step_size = delta
        self.Tolerance = tol
        self.alpha = adjust_factor  # Factor to adjust stepsize and/or initial temperature
        self.accept = 0
        self.StopCondition = False
        self.EQ = False
        self.upperbnd = AR_high
        self.lwrbnd = AR_low
        self.cooling_rate = cooling_rate
        self.interval = N_int
        self.potential = np.inf
        return


def Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd):
    T = SA.T
    delta = SA.step_size
    Xtrial = X + np.random.uniform(-delta, delta, size=len(X))

    # add limits to the parameter values:
    for i in range(len(Xtrial)):
        Xtrial[i] = min(upbnd[i], max(lwrbnd[i], Xtrial[i]))

    # Let V({dataset}|{parameterset}) be your residual function.
    # Metropolis:
    Vnew = V(xdata, ydata, yerr, Xtrial)
    Vold = SA.potential
    if (np.random.uniform() < np.exp(-(Vnew - Vold) / T)):
        X = Xtrial
        SA.accept += 1
        SA.potential = Vnew
    return X


def AcceptanceRatio(SA):
    AR = (SA.accept / float(SA.interval)) * 100
    if AR > SA.upperbnd:
        SA.step_size *= SA.alpha
    elif AR < SA.lwrbnd:
        SA.step_size /= SA.alpha
    else:
        SA.EQ = True  # <--- the next time around you'll go to TemperatureCycle()
    SA.accept = 0  # reset counter
    return


def Temperature_Cycle(SA, xdata, ydata, yerr, X, Xstart):
    update_temperature(SA)
    try:
        Enew = E
    except:
        E = V(xdata, ydata, yerr, Xstart)
        Enew = E

    # compare relative change in "equillibrium residuals".
    # If the average energy does not change more then the set tolerance between two consequetive temperatures
    # this means you are sufficiently close to the global minimum.
    E = V(xdata, ydata, yerr, X)
    SA.StopCondition = (np.abs(E - Enew) / Enew) < SA.Tolerance
    SA.EQ = False  # Reset
    return


def update_temperature(SA):
    SA.T *= SA.cooling_rate
    return





def InitialLoop(SA, X, xdata, ydata, yerr, lwrbnd, upbnd):
    '''
    Finds starting temperature for SA optimisation by performing some initial iterations until acceptance ratio
    is within acceptable bounds.
    :param SA:
    :param X:
    :param xdata:
    :param ydata:
    :param yerr:
    :param lwrbnd:
    :param upbnd:
    :return:
    '''
    steps = 0
    while True:
        steps += 1
        if (steps % SA.interval == 0):
            AR = (SA.accept / float(SA.interval)) * 100
            if AR > SA.upperbnd:
                SA.T /= SA.alpha
                SA.accept = 0
            elif AR < SA.lwrbnd:
                SA.T *= SA.alpha
                SA.accept = 0
            else:
                SA.accept = 0
                break
        X = Metropolis(SA, X, xdata, ydata, yerr, lwrbnd, upbnd)
    return
