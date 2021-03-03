# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""
import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as la
from autograd import numpy as anp
import autograd as auto
from autograd import grad
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    
    x = sy.symbols('x')
    exp = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
#    der = sy.simplify(sy.diff(exp, x))
    
#    f = sy.lambdify(x, exp)
    f1 = sy.lambdify(x, sy.diff(exp, x))
    
#    domain = np.linspace(-np.pi, np.pi, 100)  
    
#    plt.plot(domain, f(domain), label = exp)
#    plt.plot(domain, f1(domain), label = der)
#    plt.gca().spines["bottom"].set_position("zero")
#    plt.legend()
    
#    plt.show()
    return f1

    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x+h) - f(x))/h
    
    raise NotImplementedError("Problem 2 Incomplete")

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x+h) - f(x+2*h))/(2*h)
    raise NotImplementedError("Problem 2 Incomplete")

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x)-f(x-h))/h
    raise NotImplementedError("Problem 2 Incomplete")

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x-h) + f(x-2*h))/(2*h)
    raise NotImplementedError("Problem 2 Incomplete")

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h)-f(x-h))/(2*h)
    raise NotImplementedError("Problem 2 Incomplete")

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h))/(12*h)
    raise NotImplementedError("Problem 2 Incomplete")

def prob2Plots(k = 100, h = 1e-5):

    #The domain that will be used
    difD = np.linspace(-np.pi, np.pi, k)
    
    x = sy.symbols('x')
    exp = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    der = sy.simplify(sy.diff(exp, x))
    
    #Get functions
    f = sy.lambdify(x, exp)
    f1 = sy.lambdify(x, der)
    
    #Getting Y-Values
    FDQ1 = fdq1(f, difD)
    FDQ2 = fdq2(f, difD)
    BDQ1 = bdq1(f, difD)
    BDQ2 = bdq2(f, difD)
    CDQ2 = cdq2(f, difD)
    CDQ4 = cdq4(f, difD)
    
    
    #Ploting everything
    fig, graph = plt.subplots(4,2)
    
    graph[0,0].plot(difD, f(difD))
    graph[0,0].set_title(exp)
    
    graph[0,1].plot(difD, f1(difD))
    graph[0,1].set_title(der)
    
    graph[1,0].plot(difD, FDQ1)
    graph[1,0].set_title("fdq1")
    
    graph[1,1].plot(difD, FDQ2)
    graph[1,1].set_title("fdq2")
    
    graph[2,0].plot(difD, BDQ1)
    graph[2,0].set_title("bdq1")
    
    graph[2,1].plot(difD, BDQ2)
    graph[2,1].set_title("bdq2")
    
    graph[3,0].plot(difD, CDQ2)
    graph[3,0].set_title("cdq2")
    
    graph[3,1].plot(difD, CDQ4)
    graph[3,1].set_title("cdq4")
    
    plt.tight_layout()
    plt.gca().spines["bottom"].set_position("zero")
    
    plt.show()
    return

# Problem 3
def prob3(x0 = 0, K = 9):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #Code from prob1start
    x = sy.symbols('x')
    exp = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    der = sy.simplify(sy.diff(exp, x))
    
    f = sy.lambdify(x, exp)
    f1 = sy.lambdify(x, der)
    #Code from prob 1 end
    
    #just for ease of coding...
    difD = np.array(x0)
    
    #Initialize what I want
    FDQ1 = []
    FDQ2 = []
    BDQ1 = []
    BDQ2 = []
    CDQ2 = []
    CDQ4 = []
    N = []
    correct = f1(x0)
    
    #Get errors
    for i in range(0, K):
        n = 10**(-i)
        
        N.append(n)
        
        FDQ1.append(np.abs(correct - fdq1(f, difD, h = n)))
        
        FDQ2.append(np.abs(correct - fdq2(f, difD, h = n)))
        
        BDQ1.append(np.abs(correct - bdq1(f, difD, h = n)))
        
        BDQ2.append(np.abs(correct - bdq2(f, difD, h = n)))
        
        CDQ2.append(np.abs(correct - cdq2(f, difD, h = n)))
        
        CDQ4.append(np.abs(correct - cdq4(f, difD, h = n)))
        
    #Plot results
    plt.loglog(N, FDQ1, label = "Order 1 Forward")
    plt.loglog(N, FDQ2, label = "Order 2 Forward")
    plt.loglog(N, BDQ1, label = "Order 1 Backward")
    plt.loglog(N, BDQ2, label = "Order 2 Backward")
    plt.loglog(N, CDQ2, label = "Order 2 Centered")
    plt.loglog(N, CDQ4, label = "Order 4 Centered")
    
    #Tidy results
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.title("title for approximate derivative errors graphs that isn\'t in book")
    plt.legend(loc = 2)
    plt.show()
    return
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4(d = 500):
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """  
    #import the plane data
    planeData = np.load("plane.npy")
    
    tplane = planeData[:,0]
    alpha = np.deg2rad(planeData[:,1])
    beta = np.deg2rad(planeData[:,2])
    
    l = len(tplane)
    
    #define x and y functions
    def x(n):
#   Gives x position
        return d * np.tan(beta[n]) / (np.tan(beta[n]) - np.tan(alpha[n]))
    def y(n):
#   Gives y position
        return d * np.tan(beta[n]) * np.tan(alpha[n]) / (np.tan(beta[n]) - np.tan(alpha[n]))
    
    #define x and y prime as we will see them
    def xprime(n):
#   Gives the approximate derivative of x
        if n == 0:
            return fdq1(x, n, h = 1)
        elif n == l-1:
            return bdq1(x, n, h = 1)
        elif n > 0 and n < l:
            return cdq2(x, n, h = 1)
        else:
            return 0
        
    def yprime(n):
#   Gives the approximate derivative of y
        if n == 0:
            return fdq1(y, n, h = 1)
        elif n == l-1:
            return bdq1(y, n, h = 1)
        elif n > 0 and n < l:
            return cdq2(y, n, h = 1)
        else:
            return 0
        
    #define speed from x and y prime
    def speed(n):
#        print("speed(n) where n = " + str(n))
        return np.sqrt((xprime(n))**2 + (yprime(n))**2)
    
    #Finally get the speed from the information we have
    spd = []
    X = []
    Y = []
    for i in range(0, l):
        spd.append(speed(i))
        X.append(x(i))
        Y.append(y(i))
        
    return spd
    
    raise NotImplementedError("Problem 4 Incomplete")

# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    
    hn = np.zeros(np.shape(x))
    for i in range(0, len(x)):
        hn[i] = h
        #Approximate partial
        if i == 0:
            partials = ((f(x+hn)-f(x-hn))/la.norm(2*hn))
        else:
            xn = ((f(x+hn)-f(x-hn))/la.norm(2*hn))
            partials = np.vstack((partials, xn))
        hn[i] = 0
    #Return Transpose
    return partials.transpose()
    raise NotImplementedError("Problem 5 Incomplete")

def prob5(f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2]), X = np.array([2,2])):
    
    x, y = sy.symbols("x, y")
    F = sy.Matrix(f([x, y]))
    A = F.jacobian([x, y])
    C = sy.lambdify([x,y], A)
    print(C(X[0],X[1]))    
    B = jacobian_cdq2(f, X)
    print(B)
    return


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    if n == 0:
        return anp.array([1 for i in x])
    elif n == 1:
        return x
    else:
        return 2*x*cheb_poly(x, n-1)-cheb_poly(x, n-2)

    raise NotImplementedError("Problem 6 Incomplete")

def prob6(n = 5):
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = np.linspace(-1,1, 100)
    X = anp.array(list(domain), dtype = anp.float)
    dTn = auto.elementwise_grad(cheb_poly)
    primes = []
    
    for i in range(0, n):
        primes.append(dTn(X, i))
    
    #Plot graphs
    fig, graph = plt.subplots(3, 2)
    
    graph[0, 0].plot(X, primes[0], label = "T0")
    graph[0, 0].set_title("Chebyshev - T0")
    
    graph[0, 1].plot(X, primes[1], label = "T1")
    graph[0, 1].set_title("Chebyshev - T1")
    
    graph[1, 0].plot(X, primes[2], label = "T2")
    graph[1, 0].set_title("Chebyshev - T2")
    
    graph[1, 1].plot(X, primes[3], label = "T3")
    graph[1, 1].set_title("Chebyshev - T3")
    
    graph[2, 0].plot(X, primes[4], label = "T4")
    graph[2, 0].set_title("Chebyshev - T4")
    
    plt.tight_layout()
    
    return
    
#    raise NotImplementedError("Problem 6 Incomplete")

# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    #lists for approximating
    Narray = []
    p1 = []
    error1 = []
    p2 = []
    error2 = []
    p3 = []
    error3 = []
    
    x = sy.symbols('x')
    exp = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    
    fi = sy.lambdify(x, exp)
    
    for i in range(0, N):
        Narray.append(N)
        x0 = np.random.rand()
        
        start = time.time()
        
        #timing problem 1
        f1 = prob1()
        actual = f1(x0)
        
        end = time.time()
        
        p1.append(end-start)
        error1.append(1e-18)
        
        
        #Timing cdq4
        start = time.time()
        
#        def f(x): return (np.sin(x) + 1)**(np.sin(np.cos(x)))
        apprx = cdq4(fi, x0)
        
        end = time.time()
        p2.append(end-start)
        error2.append(abs(actual-apprx))
        
        
        #Timing grad
        start = time.time()
      
        #For some reason I had to put this here...
        def myfunc(x):
            return (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))
        
        df = grad(myfunc)
        apprx = df(x0)
        
        end = time.time()
        p3.append(end-start)
        error3.append(abs(apprx-actual))
    
    #Plotting stuff
    plt.loglog(p1, error1, label = "SymPy", marker = "*", linestyle = "")
    plt.loglog(p2, error2, label = "Difference Qutoients", marker = "*", linestyle = "")
    plt.loglog(p3, error3, label = "Autograd", marker = "*", linestyle = "")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    
    #This part was just to see how close I could copy the figure...
    plt.title("Figure 8.2: Solution with N = 200", y = -.25)
    
    plt.legend()
    plt.show()
    return
    raise NotImplementedError("Problem 7 Incomplete")    