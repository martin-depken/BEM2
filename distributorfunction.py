import numpy as np
import models
from time import time
from SimulatedAnnealing_Changed import chi_squared

def dist_func(xdata,ydata,yerr,params,SA):

    '''
    xdata_occ = np.array(xdata[0])
    ydata_occ = np.array(ydata[0])
    error_occ = np.array(yerr[0])
    
    xdata_on  = np.array(xdata[1])
    ydata_on  = np.array(ydata[1])
    error_on  = np.array(yerr[1])    
    
    xdata_off = np.array(xdata[2])
    ydata_off = np.array(ydata[2])
    error_off = np.array(yerr[2])
    '''
    
    # Process MP
    models_array = [models.occ_final , models.on_final , models.off_final]
    for k in range(3):
        SA.inQ.put( (xdata[k],ydata[k],yerr[k],params,models_array[k]) )
        
    residual_sum = 0.0
    for k in range(3):
        residual_sum += SA.outQ.get()   
    
    '''
    time_a = time()
    
    worker_results = [workers.apply_async(chi_squared , args = (xdata[k],ydata[k],yerr[k],params,models_array[k])) for k in range(3)]
        
    # Retrieve residuals from different processes / cores
    residual_sum = 0.0
    for w in worker_results:
        residual_sum += w.get()
    time_b = time()
    print 'Time for MP: ', time_b-time_a
    
    raw_input('Residual: ' + str(np.sum(residual_sum)))
    '''
    
    '''
    time_a = time()
    res_occ = chi_squared(xdata[0],ydata[0],yerr[0],params,models_array[0])
    res_on  = chi_squared(xdata[1],ydata[1],yerr[1],params,models_array[1])
    res_off = chi_squared(xdata[2],ydata[2],yerr[2],params,models_array[2])
    residual_sum = np.sum((res_occ,res_on,res_off))
    time_b = time()
    print 'Time for SP: ', time_b-time_a
    raw_input('Residual: ' + str(np.sum(residual_sum)))
    '''    
    return np.sum(residual_sum)
    
