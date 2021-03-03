# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

import time
from random import random
import numpy as np
from matplotlib import pyplot as plt

# Problem 1

def prob1():
    #Just testing rotating the horses so that it shouldn't appear in your test driver
    data = np.load("horse.npy")
    
    plt.plot(data[0], data[1], 'k.')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    myStretch = stretch(data, .5, 1.2)
    plt.plot(myStretch[0], myStretch[1], 'k.')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    myShear = shear(data, .5, 0)
    plt.plot(myShear[0], myShear[1], 'k.')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    
    myReflect = reflect(data, 0, 1)
    plt.plot(myReflect[0], myReflect[1], 'k.')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    myRotate = rotate(data, np.pi/2)
    plt.plot(myRotate[0], myRotate[1], 'k.')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """

    # Creating the stretch matrix
    M = np.array([[a, 0],[0, b]])

    return np.dot(M,A)
    
    raise NotImplementedError("Problem 1 Incomplete")

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """

    # Creating the shear matrix
    M = np.array([[1,a],[b,1]])
    
    return np.dot(M, A)

    raise NotImplementedError("Problem 1 Incomplete")

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """

    # Creating the reflect matrix    
    M = (1/(a**2 + b**2)) * np.array([[a**2 - b**2, 2*a*b],[2*a*b,b**2-a**2]])

    return np.dot(M, A)    

    raise NotImplementedError("Problem 1 Incomplete")

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """

    # Creating the rotate matrix    
    M = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    try:
        return np.dot(M,A)
    
    except ValueError:
        print("Matrix M is " + str(np.shape(M)) + " but matrix A is " + str(np.shape(A)))
        return None
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def prob2():
    #Tests the solar_system function to make sure the image is right
    solar_system()
    return

def solar_system(T = np.pi*3/2, omega_e = 1, omega_m = 13, x_e = 10, x_m = 11):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    #This gives me the points I'm going to graph
    myChange = np.linspace(0, T, 100)
    
    #Initializing the starting positions of the Earth and Moon
    Se = np.array([x_e,0])
    Sm = np.array([x_m,0])
    
    #This is where I'm putting my resulting position of Earth and Moon
    myCoordinatesE = []
    myCoordinatesM = []
    
    for i in range(0, len(myChange)):

        #Performing the transformation from the original position to the i'th position
        myCoordinatesE.append(rotate(Se, myChange[i] * omega_e))
        myCoordinatesM.append(rotate((Sm - Se), myChange[i] * omega_m) + myCoordinatesE[i])
    
    #Formatting for pyplot
    x = np.array(myCoordinatesE)
    m = np.array(myCoordinatesM)
    x = x.transpose()
    m = m.transpose()
    
    #Plotting the transformation
    plt.plot(x[0], x[1])
    plt.plot(m[0],m[1])

#    plt.set_aspect("equal")
    plt.axis('equal')
    plt.show()
    
    return    
    raise NotImplementedError("Problem 2 Incomplete")


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3(n = 8):
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    
    #Initializing empty lists of data that I want
    vectorTimes = []
    matrixTimes = []
    mySize = []

    for i in range(1, n+1):
        #Creating random matices and vector
        A = random_matrix(2**i)
        x = random_vector(2**i)
        B = random_matrix(2**i)
        
        #Timing matrix - vector product
        vectS = time.clock()
        matrix_vector_product(A, x)
        vectE = time.clock()
        
        #Timing matrix - matrix product
        matrixS = time.clock()
        matrix_matrix_product(A, B)
        matrixE = time.clock()
        
        #Finding the time of the products
        vectorTimes.append(vectE - vectS)
        
        matrixTimes.append(matrixE - matrixS)
        
        #recording the dimension the entries used
        mySize.append(2**i)
        
    #Plotting the data I obtained
    vectorPlot = plt.subplot(121)
    vectorPlot.plot(mySize, vectorTimes)
    plt.title("Matrix-Vector Multiplication")
    
    matrixPlot = plt.subplot(122)
    matrixPlot.plot(mySize, matrixTimes)
    plt.title("Matrix-Matrix Multiplication")
    
    plt.show()
    
    return
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def prob4(n = 8):
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    
    #Initializing the data I want to collect
    vectorTimes = []
    matrixTimes = []
    dotVectorTimes = []
    dotMatrixTimes = []
    mySize = []
    
    for i in range(1, n+1):
        
        #Initializing random matrices and vectors
        A = random_matrix(2**i)
        x = random_vector(2**i)
        B = random_matrix(2**i)
        
        #timing vector list product
        vectS = time.clock()
        matrix_vector_product(A, x)
        vectE = time.clock()
        
        #timing matrix list product
        matrixS = time.clock()
        matrix_matrix_product(A, B)
        matrixE = time.clock()
        
        #timing pythons matrix matrix product
        matrixDS = time.clock()
        np.dot(A,B)
        matrixDE = time.clock()
        
        #timing pythons matrix vector product
        vectDS = time.clock()
        np.dot(A, x)
        vectDE = time.clock()
        
        #recording the times
        dotVectorTimes.append(vectDE-vectDS)
        
        dotMatrixTimes.append(matrixDE - matrixDS)
        
        vectorTimes.append(vectE - vectS)
        
        matrixTimes.append(matrixE - matrixS)
    
        #recording the dimension at those times
        mySize.append(2**i)
        
    #Plotting regular times
    regularPlot = plt.subplot(121)
    regularPlot.plot(mySize, vectorTimes, label = "vector x vector")
    regularPlot.plot(mySize, matrixTimes, label = "matrix x vector")
    regularPlot.plot(mySize, dotVectorTimes, label = "vector * vector")
    regularPlot.plot(mySize, dotMatrixTimes, label = "matrix * vector")
    plt.legend()
    plt.title("Not Log Based Plot")
    
    #Plotting log based times
    logPlot = plt.subplot(122)
    logPlot.loglog(mySize, vectorTimes, basex = 2, basey = 2, label = "vector x vector")
    logPlot.loglog(mySize, matrixTimes, basex = 2, basey = 2, label = "martix x vector")
    logPlot.loglog(mySize, dotVectorTimes, basex = 2, basey = 2, label = "vector * vector")
    logPlot.loglog(mySize, dotMatrixTimes, basex = 2, basey = 2, label = "matrix * vector")
    plt.title("Log Based Plot")
    plt.legend()
    plt.show()    
    
    return
    raise NotImplementedError("Problem 4 Incomplete")
