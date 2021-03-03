# newtons_method.py
"""Volume 1: Newton's Method.
<Name>
<Class>
<Date>
"""
import sympy as sy
from autograd import grad
import numpy as np
from autograd import jacobian
import autograd.numpy as anp
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    
    #Do I need to use autograd or np? 
    #Should I convert one to the other?
    #Set initial values
    diff = 1
    it = 0
    myBool = True
    
    #While value outside tolerance

    while diff > tol:
        #Check max iterations not reached
        if maxiter <= it:
            myBool = False
            break
        
        #Get next guess
        der = Df(x0)
#        print(der)
        if np.isscalar(der):
            x1 = x0 - alpha * f(x0)/der
        else:
            myIn = np.linalg.inv(der)
            x1 = x0 - alpha * myIn @ f(x0)
        
        #Check if new guess is good
        if np.isscalar(x0):
            diff = np.abs(x0 - x1)
        else:
            diff = np.linalg.norm(x0-x1)
        x0 = x1
        it += 1
        
    #return results
    return x0, myBool, it
    
    raise NotImplementedError("Problem 1 Incomplete")

#Testing functions
def prob1(f = lambda x: np.e**x - 1, x0 = 10, df = lambda x: np.e**x):
    return newton(f, x0, df)

def prob3(a = .4):
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1/3)
    df = grad(f)
    x0 = .01
    return newton(f, x0, df, alpha = a)

def prob4(k = 99):
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1/3)
    df = grad(f)
    x0 = .01
    return optimal_alpha(f, x0, df, n = k)

def prob5(x0 = np.array([1,0])):
#    f = lambda x, y: anp.array([(x-y) * np.cos(y), (x-y) * anp.sin(y)])
    
    return newton(myF, x0, myDF)

def myF(x):
    return np.array([(x[0]-x[1])*np.cos(x[1]), (x[0]-x[1])*np.sin(x[1])])

def myDF(x):
    h = 1e-7
    return np.array([(myF(x+h*np.array([1,0]))-myF(x))/h, (myF(x+h*np.array([0,1]))-myF(x))/h])

    
# Problem 2
def prob2(N1, N2, P1, P2, r0 = .1):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #Convert f(a,b,c,d, r) = 0 to F(r) = 0
    f = lambda x: P1*((1 + x)**N1 - 1) - P2*(1 - (1 + x)**(-N2))
    
    #Get derivative
    df = grad(f)
    
    #Return converge point (assuming it converges)
    return newton(f, r0, df)[0]
    
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15, n = 99):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #Get domain of alpha values
    domain = np.linspace(1/(n+1), 1, int(n))
    
    #Initialize a couple of lists and optimal values
    xval = []
    iters = []
    minit = maxiter
    bestAlpha = 1
    
    for i in domain:
        x, b, it = newton(f, x0, Df, tol, maxiter, i)
        if b:
            xval.append(i)
            iters.append(it)
            if minit > it:
                minit = it
                bestAlpha = i
    #Plot stuff
    plt.plot(xval, iters, label = "alpha x Iterations")
    plt.xlabel("Alpha")
    plt.ylabel("Iterations")
    #I don't know if it's a valley or an arrow or what this shape is
    plt.title("Abbreviated title (see comment): IDKIIAVOAAOWTSI")
    plt.legend()
    plt.show()
    
    return bestAlpha
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 6
def prob6(N = 50):
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
#    try:
#    5*x*y - x*(1+y) = 0
#    -(x*y) + (1-y**2) = 0
    #Get domain
    D = np.linspace(1/int(np.sqrt(N)), .25, int(np.sqrt(N))-1)
    #Iterate through various x, y points
    for i in D:
        for j in D:
            x0 = np.array([-i, j])
            a = -i -1
            b = 12*j**2/(2*j-1)
            if a != b:                
                #Get the newton decomposition
                res = newton(Bio6, x0, DBio6, alpha = 1, maxiter = 20)
                
                #Check that it's valid
                if res[1]:
                    if np.allclose(res[0], np.array([0,1])) or np.allclose(res[0], np.array([0,-1])):
                        res = newton(Bio6, x0, DBio6, alpha = .55, maxiter = 20)
                        if res[1]:
                            if np.allclose(res[0], np.array([3.75, .25])):
                                return x0
    #Failed to find it.
    return "Failed"
#    except LinAlgError:
#        print("Singular: " + str(x0))
    #return
    raise NotImplementedError("Problem 6 Incomplete")
def prob6test(x0):
    res = newton(Bio6, x0, DBio6, alpha = 1, maxiter = 20)
    #Check that it's valid
    if res[1]:
        if np.allclose(res[0], np.array([0,1]), atol = 1e-6) or np.allclose(res[0], np.array([0,-1]), atol = 1e-6):
            res = newton(Bio6, x0, DBio6, alpha = .55, maxiter = 20)
            if res[1]:
                if np.allclose(res[0], np.array([3.75, .25]), atol = 1e-6):
                    return x0
                else:
                    print(x0)
    
def Bio6(x):
    return np.array([4*x[0]*x[1] - x[0], -x[0]*x[1] + (1-x[1]**2)])


def DBio6(x):
    return np.array([[4*x[1] - 1, 4*x[0]],[-x[1], -x[0]-2*x[1]]])


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    #Make domain
    x_r = np.linspace(domain[0], domain[1], res)
    x_i = np.linspace(domain[2], domain[3], res)
    x_r, x_i = np.meshgrid(x_r, x_i)
    x_i = 1j*x_i
    X = x_r + x_i
    
    #Find everything
    Y = []
    for i in X:
        result = []
        for j in i:      
            result.append(np.argmin(np.abs(newtit(f, Df, j, iters) - zeros)))
        Y.append(result)

#    print(Y)
    
    #Plot everythin
    plt.pcolormesh(x_r, x_i/1j, Y, cmap = "brg")    
    plt.title("Colorful")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.show()
    return
    
    raise NotImplementedError("Problem 7 Incomplete")
    
def f71(x):
    return x**3 - 1

def f72(x):
    return x**3 - x

def df71(x):
    return 3*x**2

def df72(x):
    return 3*x**2-1

def prob7(rez = 1000, f = f71, df = df71, zeros = np.array([1, -1/2 + 1j*np.sqrt(3/4), -1/2 - 1j * np.sqrt(3/4)]), dom = np.array([-1.5,1.5,-1.5,1.5])):
    plot_basins(f, df, zeros, dom, res = rez)
    
def prob72():
    plot_basins(f72, df72, np.array([0, 1, -1]), np.array([-1.5,1.5,-1.5,1.5]))
    return
def newtit(f, Df, x0, iters = 15):
    for i in range(0, iters):
        x0 = x0 - f(x0)/Df(x0)
    return x0
