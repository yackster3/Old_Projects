# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import stats as stat
from matplotlib import pyplot as plt

def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #get my points
    points = np.random.uniform(-1, 1, (n, N))
    
    #get my ins and outs
    lengths = la.norm(points, axis=0)
    numWithin = np.count_nonzero(lengths < 1)
    
    #return result
    return (2**n) * (numWithin/N)

def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #get points
    points = np.random.uniform(a, b, (1, N))
    
    #get vals
    yvals = f(points[0])
    
    #get results
    return np.mean(yvals)*(b-a)
    

def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """

    #Get samples
    samples = np.random.uniform(0, 1, (N, len(mins)))
    scale = samples * (maxs-mins)
    points = scale + mins
    
    #get ins and outs
    yvals = 0
    for pt in points:
        yvals += f(pt) 
    yvals = yvals/N
    
    #get total volume
    Vol = np.product(maxs - mins)
    
    #get result
    return np.mean(yvals)*Vol
    
    raise NotImplementedError("Problem 3 Incomplete")

def prob3(mins = np.array([0, 0]), maxs = np.array([1, 1]), f = lambda x: x[0]**2 + x[1]**2, N = 10000):
    return mc_integrate(f, mins, maxs, N)

def prob3test2(N = 10**5):
    f = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    mins = np.array([1, -2])
    maxs = np.array([3, 1])
    return mc_integrate(f, mins, maxs, N) 

def prob3test3(N = 10**7):
    f = lambda x: x[0] + x[1] + x[2] * x[3]**2
    mins = np.array([-1, -2, -3, -4])
    maxs = np.array([1, 2, 3, 4])
    return mc_integrate(f, mins, maxs, N)

def prob4(mins = np.array([-3/2, 0, 0, 0]), maxs = np.array([3/4, 1, 1/2, 1])):
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #make function
    f = lambda x: (np.e**(-(la.norm(x)**2)/2))/((2*np.pi)**(len(x)/2))
    
    #get means and covariance
    means, cov = np.zeros(len(mins)), np.eye(len(mins))
    Istat = stat.mvn.mvnun(mins, maxs, means, cov)[0]
    
    #get logspace
    N = np.logspace(1, 5, num = 20, dtype = int)
    Imy = []
    error = []
    sqrt = []
    
    #Get relative error
    for i in N:
        Imy.append(mc_integrate(f, mins, maxs, i))
        error.append(np.abs(Istat - Imy[-1])/np.abs(Istat))
        sqrt.append(1/np.sqrt(i))
    
    #plot results
    plt.loglog(N, error, label = "Relative Error")
    plt.loglog(N, sqrt, label = "1/sqrt(N)")
    plt.legend()
    plt.show()
    
    return
