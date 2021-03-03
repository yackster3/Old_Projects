# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""
from matplotlib import pyplot as plt
import sympy as sy
import numpy as np
from scipy import linalg as la
from scipy.interpolate import BarycentricInterpolator as Bary
# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    #if there are the incorrect number of xint and yints
    if len(xint) != len(yint):
        raise ValueError("invalid X and Y values")
    
    #make polynomial variable
    x = sy.symbols('x')
    l = len(xint)
    L = []
    #lagrange interpolation
    for j in range(0, l):
        
        #delete first term
        y = xint[0]
        xint = np.delete(xint, 0)
        
        #get the top and bottom terms
        expT = np.product(x - xint)
        expB = np.product(y - xint)

        #append to the end
        xint = np.append(xint, y)
        
        #append the expression to L
        L.append(expT/expB)
    
    #convert L to array (for consistency)
    L = np.array(L)
    
    #make expressions with yint
    exp = np.dot(L, yint)
    
    #make the expressions functions
    F = sy.lambdify(x, exp)
    
    return F(points)
    raise NotImplementedError("Problems 1 and 2 Incomplete")


def prob1(n = 5):
    
    #evenly spaced points
    xint = np.linspace(-1, 1, n)
    
    #make test function
    x = sy.symbols('x')
    f = 1/(1+(5*x)**2)
    F = sy.lambdify(x, f)
    
    #get y values
    yint = F(xint)
    
    #plot points
    points = np.linspace(-1, 1, 100)
    
    plt.plot(points, F(points), label = "f(x) = 1/(1+25*x^2)")
    plt.plot(points, lagrange(xint, yint, points), label = "lagrange")
    plt.legend()
    plt.show()
    
    return
# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.xint = xint
        self.yint = yint
        
        self.w = []
        
        l = len(xint)
        for j in range(0, l):
        
            #delete first term
            y = xint[0]
            xint = np.delete(xint, 0)
            
            #get the top and bottom terms
            expB = np.product(y - xint)
            
            #append to the end
            xint = np.append(xint, y)
            
            #append the expression to w
            self.w.append(1/expB)
        
        return 
        raise NotImplementedError("Problem 3 Incomplete")

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        ans = []
        for i in points:
            
            #just put in the yval if the point is in xint
            if i in self.xint:
                np.delete(points, i)
                ans.append(self.yint[np.where(self.xint == i)])
            
            #evaluate function at points
            else:
                ans.append(sum(self.w*self.yint/(i-self.xint))/sum(self.w/(i-self.xint)))
    
        return np.array(ans)
        
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        if len(xint) != len(yint):
            raise ValueError("Invalid xint/yint, not the same length")
        i = 0
        
        #while loop works better than for loop for some
        #reason when getting rid of duplicate weights
        while i < len(xint):
            if xint[i] in self.xint:
                xint = np.delete(xint, i)
                yint = np.delete(yint, i)
            else:
                i += 1   
                
        #add new weights
        for i in range(0, len(xint)):
            nextw = 1/(self.xint - xint[i])
            self.w = self.w * nextw
            
            self.w = np.append(self.w, 1/np.product(xint[i]-self.xint))
            self.xint = np.append(self.xint, xint[i])
            self.yint = np.append(self.yint, yint[i])
        return
        raise NotImplementedError("Problem 4 Incomplete")

def prob3(n = 5):
    #evenly spaced points
    xint = np.linspace(-1, 1, n)
    
    #make test function
    x = sy.symbols('x')
    f = 1/(1+(5*x)**2)
    F = sy.lambdify(x, f)
    
    #get y values
    yint = F(xint)
    
    #Barycentric
    b = Barycentric(xint, yint)
    
    #plot points
    points = np.linspace(-1, 1, 100)
    
    yvals = b.__call__(points)
    
    plt.plot(points, F(points), label = "f(x) = 1/(1+25*x^2)")
    plt.plot(points, lagrange(xint, yint, points), label = "lagrange")
    plt.plot(points, yvals, label = "barycentric")
    plt.legend()
    plt.show()

def prob4(n = 3):
    
    xint1 = np.linspace(-1, 1, n)
    xint2 =  np.linspace(-1, 1, n**2)
    
    #make test function
    x = sy.symbols('x')
    f = 1/(1+(5*x)**2)
    F = sy.lambdify(x, f)
    
    #get y values
    yint1 = F(xint1)
    yint2 = F(xint2)
    
    #Barycentric getting weights
    b = Barycentric(xint1, yint1)
    b.add_weights(xint2, yint2)
    c = Barycentric(xint1, yint1)
    
    #plot points
    points = np.linspace(-1, 1, 100)
    
    plt.plot(points, F(points), label = "f(x) = 1/(1+25*x^2)")
    
    #plt.plot(points, lagrange(xint1, yint1, points), label = "lagrange")
    plt.plot(points, b.__call__(points), label = "barycentric")
    plt.plot(points, c.__call__(points), label = "barycentric - 2")
    plt.legend()
    plt.ylim(-1, 2)
    plt.xlim(-1, 1)
    plt.show()
    
    
    
# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    f = lambda x: 1/(1+25*x**2) #function to be interpolated
    #obtain the chebyshev etremal points on [-1,  1]
    
    domain = np.linspace(-1, 1, 200)

    #Initalize lists
    errEven = []
    errCheby = []
    n = []
    
    #Obtain the error of the 
    #even and chebyshev points
    for i in range(2, 9):
        n.append(2**i)
        pts = np.linspace(-1, 1, 2**i)
        poly = Bary(pts)
        poly.set_yi(f(pts))
        
        errEven.append(la.norm(f(domain) - poly(domain), ord = np.inf))
        
        pts = [np.cos(j*np.pi/n[-1]) for j in range(0, n[-1] + 1)]
        pts = np.array(pts)
        
        poly = Bary(pts)
        poly.set_yi(f(pts))
        
        errCheby.append(la.norm(f(domain) - poly(domain), ord = np.inf))
    
    #plot stuff with log scale x-axis base 2
    plt.loglog(n, errEven, label = "Even distribution")
    plt.loglog(n, errCheby, label = "Chebyshev distribution")
    plt.xscale('log', basex = 2)
    plt.legend()
    plt.show()
    
    return 
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    #Get samples
    y = np.cos((np.pi * np.arange(2*n))/n)
    samples = f(y)
    
    #FFT of samples
    coeffs = np.real(np.fft.fft(samples))[:n+1]/n
    
    #tidy up
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2
    
    #return
    return coeffs
    
    raise NotImplementedError("Problem 6 Incomplete")

def prob6(n = 4):
    f = lambda x: -3 + 2*x**2 - x**3 + x**4
    pcoeffs = [-3,0,2,-1,1]
    ccoeffs = np.polynomial.chebyshev.poly2cheb(pcoeffs)
    fpoly = np.polynomial.Polynomial(pcoeffs)
    fcheb = np.polynomial.Chebyshev(ccoeffs)
    
    answer = chebyshev_coeffs(f, n)
    
    print("fpoly: " + str(fpoly))
    print("fcheb: " + str(fcheb))
    print("answer: " + str(answer))
    return
    
# Problem 7
def prob7(n = 50):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    data = np.load('airdata.npy')
 
    #Code I was given begin
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1)*np.pi/n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a,b,n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis = 0)
    #Code I was given end
    
    #get rid of duplicate indices
    temp2 = set(temp2)
    temp2 = list(temp2)
    
    #Interpolate temp2 are the indeces of my chebychev points in the domain
    poly = Bary(domain[temp2], data[temp2])
    
    #Plot the points
    plt.scatter(domain, data, label = "True Data", color = "green")
    plt.plot(domain, poly(domain), label = "n = " + str(n), color = "red")
    plt.ylim(min(data), max(data))
    plt.legend()
    plt.show()
    
    
    
    return
    raise NotImplementedError("Problem 7 Incomplete")
