import numpy as np
from scipy import sparse
import time
from threading import Thread
from matplotlib import pyplot as plt
from poisson import poisson
from check_matvec import check_matvec
import sys
from pdb import set_trace

def L2norm(e, h):
    '''
    Take L2-norm of e
    '''
    # ensure e has a compatible shape for taking a dot-product
    e = e.reshape(-1,) 

    # Task:
    # Return the L2-norm, i.e., the square root of the integral of e^2
    # Assume a uniform grid in x and y, and apply the midpoint rule.
    # Assume that each grid point represents the midpoint of an equally sized region
    
    l2_norm_squared = np.sum(e**2)
    
    l2_norm = h * np.sqrt(l2_norm_squared)
    
    
    return l2_norm


def compute_fd(n, nt, k, f, fpp_num):
    '''
    Compute the numeric second derivative of the function 'f' with a 
    threaded matrix-vector multiply.  

    Input
    -----
    n   <int>       :   Number of grid points in x and y for global problem
    nt  <int>       :   Number of threads
    k   <int>       :   My thread number
    f   <func>      :   Function to take second derivative of
    fpp_num <array> :   Global array of size n**2


    Output
    ------
    fpp_num will have this thread's local portion of the second derivative
    written into it


    Notes
    -----
    We do a 1D domain decomposition.  Each thread 'owns' the k*(n/nt) : (k+1)*(n/nt) rows
    of the domain.  
    
    For example, 
    Let the global points in the x-dimension be [0, 0.33, 0.66, 1.0] 
    Let the global points in the y-dimension be [0, 0.33, 0.66, 1.0] 
    Let the number of threads be two (nt=2)
    
    Then for the k=0 case (for the 0th thread), the domain rows  'owned' are
    y = 0,    and x = [0, 0.33, 0.66, 1.0]
    y = 0.33, and x = [0, 0.33, 0.66, 1.0]
    
    Then for the k = 1, case, the domain rows 'owned' are
    y = 0.66, and x = [0, 0.33, 0.66, 1.0]
    y = 1.0,  and x = [0, 0.33, 0.66, 1.0]

    We assume that n/nt divides evenly.

    '''
    # set_trace()
    # Task: 
    # Compute start, end
    #
    # These indices allow you to index into arrays and grab only this thread's
    # portion.  For example, using the y = [0, 0.33, 0.66, 1.0] example above,
    # and considering thread 0, will yield start = 0 and end = 2, so that 
    # y[start:end] --> [0, 0.33]
    start = int(k*(n/nt))
            # MAKE sure to cast start as an integer, start = int(...)
    end = int((1+k)*(n/nt))
            # MAKE sure to cast end as an integer, end = int(...)
    
    # Task:
    # Compute start_halo, and end_halo
    #
    # These values are the same as start and end, only they are expanded to 
    # include the halo region.
    #
    # Halo regions essentially expand a thread's local domain to include enough
    # information from neighboring threads to carry out the needed computation.
    # For the above example, that means
    #   - Including the y=0.66 row of points for the k=0 case
    #     so that y[start_halo : end_halo] --> [0, 0.33, 0.66]
    #   - Including the y=0.33 row of points for the k=1 case
    #     so that y[start_halo : end_halo] --> [0.33, 0.66, 1.0]
    #   - Note that for larger numbers of threads, some threads 
    #     will have halo regions including domain rows above and below.
    start_halo = start - 1 if k!=0 else start
    end_halo = end + 1 if k!= (nt-1) else end
    
    # Construct local CSR matrix.  Here, you're given that function in poisson.py
    # This matrix will contain the extra halo domain rows
    A = poisson((end_halo - start_halo, n), format='csr')
    h = 1./(n-1)
    A *= 1/h**2

    # Task:
    # Inspect a row or two of A, and verify that it's the correct 5 point stencil
    # You can print a few rows of A, with print(A[k,:])
    #sys.stderr.write(A[k,:])
    # print(A[k,:])

    # Task:
    # Construct a grid of evenly spaced points over this thread's halo region
    #
    # x_pts contains all of the points in the x-direction in this thread's halo region
    x_pts = np.linspace(0,1,n)
    #
    # y_pts contains all of the points in the y-direction for this thread's halo region
    # For the above example and thread 1 (k=1), this is y_pts = [0.33, 0.66, 1.0]
    y_pts = np.linspace(start_halo*h,(end_halo-1)*h,end_halo-start_halo)
                        
    # Task:
    # There is no coding to do here, but examime how meshgrid works and 
    # understand how it gives you the correct uniform grid.
    X,Y = np.meshgrid(x_pts, y_pts)
    X = X.reshape(-1,)
    Y = Y.reshape(-1,) 

    # Task:
    # Compute local portion of f by using X and Y
    f_vals = f(X,Y)
    
    # Task:
    # Compute the correct range of output values for this thread
    output = A*f_vals
    
    # Task:
    # Set the output array
    if(nt==1):
        fpp_num[:] = output[:]
    elif (k == 0):
        fpp_num[start*n:end*n] = output[:-n]
    elif (k == nt - 1):
        fpp_num[start*n:end*n] = output[n:]
    else:
        fpp_num[start*n:end*n] = output[n:-n]
    # set_trace()
    
    
def fcn(x,y):
    '''
    This is the function we are studying
    '''
    return np.cos((x+1)**(1./3.) + (y+1)**(1./3.)) + np.sin((x+1)**(1./3.) + (y+1)**(1./3.))

# def fcnpp(x,y):
#     '''
#     This is the second derivative of the function we are studying
#     '''
#     # Task:
#     # Fill this function in with the correct second derivative.  You end up with terms like
#     # -cos((x+1)**(1./3.) + (y+1)**(1./3.))*(1./9.)*(x+1)**(-4./3)
#     term1 = -np.cos((x+1)**(1./3.) + (y+1)**(1./3.))*(1./9.)*(x+1)**(-4./3)
#     term2 = -np.sin((x+1)**(1./3.) + (y+1)**(1./3.))*(1./9.)*(x+1)**(-4./3)
#     term3 = -np.cos((y+1)**(1./3.) + (x+1)**(1./3.))*(1./9.)*(y+1)**(-4./3)
#     term4 = -np.sin((y+1)**(1./3.) + (x+1)**(1./3.))*(1./9.)*(y+1)**(-4./3)
#     return term1 + term2 + term3 + term4

def fcnpp(x, y):
    """
    Function returns the analytical second derivative with respect to the specified variable
    :param x: float point    :param y:
    :param variable: string specifying either x or y, which determines the derivative with respect to that variable
    :return: Value of the second derivative, assuming valid inputs above
    """
    argument = (x + 1) ** (1 / 3) + (y + 1) ** (1 / 3)
    var_coeff1 = x + 1
    var_coeff2 = y + 1


    p1 = 2 / 9 * var_coeff1 ** (-5 / 3) * np.sin(argument)
    p2 = -1 / 9 * var_coeff1 ** (-4 / 3) * np.cos(argument)
    p3 = -2 / 9 * var_coeff1 ** (-5 / 3) * np.cos(argument)
    p4 = -1 / 9 * var_coeff1 ** (-4 / 3) * np.sin(argument)
    p5 = 2 / 9 * var_coeff2 ** (-5 / 3) * np.sin(argument)
    p6 = -1 / 9 * var_coeff2 ** (-4 / 3) * np.cos(argument)
    p7 = -2 / 9 * var_coeff2 ** (-5 / 3) * np.cos(argument)
    p8 = -1 / 9 * var_coeff2 ** (-4 / 3) * np.sin(argument)
    
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8



##
# Here are three problem size options for running.  The instructor has chosen these
# for you.
option = 1
if option == 1:
    # Choose this if doing a final run on CARC for your strong scaling study
    NN = np.array([840*6]) # array of grid sizes "n" to loop over
    num_threads = [1,2,3,4,5,6,7,8]
elif option == 2:
    # Choose this for printing convergence plots on your laptop/lab machine,
    # and for initial runs on CARC.  
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = 210*np.arange(1,6) # array of grid sizes "n" to loop over
    num_threads = [1,2,3] #eventually include 2, 3
elif option == 3:
    # Choose this for code development and debugging on your laptop/lab machine
    # You may want to start with just num_threads=[1] and debug the serial case first.
    NN = np.array([6]) # array of grid sizes "n" to loop over
    num_threads = [1] #eventually include 2,3
elif option == 4:
    # use this for testing strong scaling plots on local machine
    NN = 210*np.array([6])
    num_threads = [1,2,3,4]
else:
    print("Incorrect Option!")

##
# Begin main computation loop
##

# Task:
# Initialize your data arrays
error = np.zeros((len(num_threads),len(NN)))
timings = np.zeros((len(num_threads),len(NN)))


# Loop over various numbers of threads
# set_trace()
for i,nt in enumerate(num_threads):
    # Loop over various problem sizes 
    for j,n in enumerate(NN):
        
        # Task:
        # Initialize output array
        fpp_numeric = np.zeros(n**2)
        
        # Task:
        # Choose the number of timings to do for each run 
        ntimings = 5

        # Carry out timing experiment 
        min_time = 10000
        for m in range(ntimings):

            # This loop will set up each Thread object to compute fpp numerically in the 
            # interior of each thread's domain.  That is, after this loop 
            # t_list = [ Thread_object_1, Thread_object_2, ...]
            # where each Thread_object will be ready to compute one thread's contribution 
            # to fpp_numeric.  The threads are launched below.
            t_list = []
            # set_trace()
            for k in range(nt):
                # Task:
                # Finish this call to Thread(), passing in the correct target and arguments
                t_list.append(Thread(target=compute_fd, args=(n, nt, k, fcn, fpp_numeric) )) 

            start = time.perf_counter()
            # Task:
            # Loop over each thread object to launch them.  Then separately loop over each 
            # thread object to join the threads.
            for thread in t_list:
                thread.start()
            for thread in t_list:
                thread.join()
            end = time.perf_counter()
            min_time = min(end-start, min_time)
        ##
        # End loop over timings
        # print(" ")

        ##
        # Use testing-harness to make sure your threaded matvec works
        # This call should print zero (or a numerically zero value)
        if option == 2 or option == 3:
            check_matvec(fpp_numeric, n, fcn)
        
        # Construct grid of evenly spaced points for a reference evaluation of
        # the double derivative
        h = 1./(n-1)
        pts = np.linspace(0,1,n)
        X,Y = np.meshgrid(pts, pts)
        X = X.reshape(-1,)
        Y = Y.reshape(-1,) 
        fpp = fcnpp(X,Y)

        # Account for domain boundaries.  
        #
        # The boundary_points array is a Boolean array, that acts like a
        # mask on an array.  For example if boundary_points is True at 10
        # points and False at 90 points, then x[boundary_points] will be a
        # length 10 array at those 10 True locations
        boundary_points = (Y == 0)
        fpp_numeric[boundary_points] += (1/h**2)*fcn(X[boundary_points], Y[boundary_points]-h)
        # Task:
        # Account for the domain boundaries at Y == 1, X == 0, X == 1
        boundary_points  = (Y==1)
        fpp_numeric[boundary_points] += (1/h**2) * fcn(X[boundary_points], Y[boundary_points]+h)
        boundary_points  = (X==0)
        fpp_numeric[boundary_points] += (1/h**2) * fcn(X[boundary_points] -h , Y[boundary_points])
        boundary_points  = (X==1)
        fpp_numeric[boundary_points] += (1/h**2) * fcn(X[boundary_points] +h , Y[boundary_points])


        
        # Task:
        # Compute error
        e = fpp_numeric - fpp 
        error[i,j] = L2norm(e, h)
        timings[i,j] = min_time
        # print(min_time)
    ##
    # End Loop over various grid-sizes
    # print(" ")
    
    # Task:
    # Generate and save plot showing convergence for this thread number
    # --> Comment out plotting before running on CARC
    # plt.figure(1)
    # plt.loglog(NN, error[i, :], label='Error')
    # plt.loglog(NN, 1/NN**2, label='Reference')
    # plt.xlabel('Grid Size (n)')
    # plt.ylabel('Error')
    # plt.legend()
    # plt.savefig('error' + str(i) + '.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)
plt.figure(1)
plt.plot(num_threads, np.transpose(timings)[-1])
plt.xlabel('Threads')
plt.xticks(range(1,len(num_threads)+1))
plt.ylabel('Time (s)')
plt.savefig('Strong_scaling_timing' + '.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)

def efficiency(timings):
    E_p = np.zeros(len(timings))
    for i in range(len(timings)):
        E_p[i] = timings[0]/(timings[i]*(i+1))
    return E_p
    
    
plt.figure(2)
plt.plot(num_threads, efficiency(np.transpose(timings)[-1]))
plt.xlabel('Threads')
plt.xticks(range(1,len(num_threads)+1))
plt.ylabel('Efficiency')
plt.savefig('Strong_scaling_efficiency' + '.png', dpi=500, format='png', bbox_inches='tight', pad_inches=0.0)

# Save timings for future use

# Save timings for future use

np.savetxt('timings.txt', timings)

