# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
import time
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
import scipy

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    B = A.copy()
    
    for i in range(0, len(A)):
        c = B[i][i] 
        if c == 0:
            raise ValueError("The matrix given is not invertable")

        #Subtracts the non-zero elements row from row
        for j in range(i+1, len(A[i])):
            k = B[j][i]
            m = k/c
            B[j,1:] = B[j,1:] - B[i,1:] * m
        
        #Fills in zero under the diagnal
        for k in range(0, i):
            B[i][k] = 0
    
    return B
    
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #WARNING
    #If np.arrays of int are stored as int this may FAIL
    #copying and setting up the upper and lower matrix
    m, n = np.shape(A)
    U = A.copy()
    L = np.eye(m)
    
    #Included in a try block for debugging, but will still catch errors later
    try:
        #2 for loops to iterate through the entries of the matrix
        for j in range(0, int(n)):   
            for i in range(j+1, int(m)):
                #making the lower and upper matrix, using l to find u
                L[i][j] = U[i][j]/U[j][j]
                U[i][j:] = U[i][j:] - L[i][j]*U[j][j:]
    except ValueError:
    #printing information to find errors
        print(j)
        print(i)
        print(L)
        print(U)
    #returning the matrix
    print(L.dot(U))
    return L, U
    
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L, U = lu(A)
    y = b.copy()
    size = len(L)
    
    #solving for y
    for i in range(0, size):
        sub = 0
        for j in range(0, i):
            sub = sub + L[i][j]*y[j]
        
        y[i] = b[i] - sub
    #since x is changing we copy y to x...probably
    #didn't really need to do this but it just helps
    #it make sense from the paper...
    x = y.copy()
    #solving for x
    for i in range(1, size+1):
        #iterating backwards
        k = size - i
        sub = 0
        for j in range(k+1, size):
            sub = sub + U[k][j] * x[j]
            
        x[k] = (x[k] - sub)/U[k][k]
    
    return x
    raise NotImplementedError("Problem 3 Incomplete")

#Functions that do parts of solve...for testing purposes
def solvePartA(A, b):
    L, U = lu(A)
    y = b.copy()
    size = len(L)
    
    #solving for y
    for i in range(0, size):
        sub = 0
        for j in range(0, i):
            sub = sub + L[i][j]*y[j]
        
        pls = b[i]-sub
        y[i] = pls
    return y

#Functions that do parts of solve...for testing purposes
def solvePartB(A, y):
    L, U = lu(A)
    size = len(L)
    x = y.copy()
    #solving for x
    for i in range(1, size+1):
        #iterating backwards
        k = size - i
        sub = 0
        for j in range(k+1, size):
            sub = sub + U[k][j] * x[j]
            
        x[k] = (x[k] - sub)/U[k][k]
    
    return x
# Problem 4
def prob4(n = 10):
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    lu = []
    slv = []
    lslv = []
    inv = []
    myX = []
    start = 0
    end = 0
    for i in range(0, n):
        myX.append(2**i)
        A = np.random.random((2**i, 2**i))
        b = np.random.random(2**i)
        
        #solving with inverse
        start = time.clock()
        
        Ainv = la.inv(A)
        x = Ainv @ b
        
        end = time.clock()
        
        inv.append(end - start)
        
        #solving with la_solve and la.lu_factor
        start = time.clock()
        
        L, P = la.lu_factor(A)
        x = la.lu_solve((L, P), b)
        
        end = time.clock()
        
        lu.append(end - start)
        
        #solving with la_solve only
        start = time.clock()
        
        x = la.solve(A, b)
        
        end = time.clock()
        
        slv.append(end - start)
        
        #solving with la_solve and la.lu_factor, only timing la.lu
        
        L, P = la.lu_factor(A)
        start = time.clock()
        x = la.lu_solve((L, P), b)
        end = time.clock()
        lslv.append(end - start)

    #I didn't know which graphs y'all would prefer...
    #Or how big you'd want the matrices that are tested...
    #...so I used both and let you put in the largest size, 
    # incrementing the matrix size by 2^n

    myPlt = plt.subplot(121)
    myPlt.loglog(myX, inv, basex = 2, basey = 2, label = "Inverse")
    myPlt.loglog(myX, slv, basex = 2, basey = 2, label = "la.solve")
    myPlt.loglog(myX, lu, basex = 2, basey = 2, label = "lu_factor")
    myPlt.loglog(myX, lslv, basex = 2, basey = 2, label = "lu_solve")
    
    plt.title("Plots of time for solving (log base 2)")
    plt.legend()
    
    myPlt = plt.subplot(122)
    myPlt.plot(myX, inv, label = "Inverse")
    myPlt.plot(myX, slv, label = "la.solve")
    myPlt.plot(myX, lu, label = "lu_factor")
    myPlt.plot(myX, lslv, label = "lu_solve")
    
    plt.title("Plots of time for solving")
    plt.legend()
    
    plt.show()
    
    return
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #Creating matrix B
    offsets = [-1,0,1]
    diagonals = [[],[],[]]
    for i in range(0, n):
        diagonals[0].append(1)
        diagonals[1].append(-4)
        diagonals[2].append(1)    
    B = sparse.diags(diagonals, offsets, shape = (n,n))
    I = np.eye(n)
    x = []
    #Creating the matrix to be returned
    for i in range(0, n):
        y = []
        #Creating the row that will go down the diagonals
        for j in range(0, n):
            if j == i-1 or j == i+1:
                y.append(I)
            elif j == i:
                y.append(B)
            else:
                y.append(None)
        x.append(y)
    
    #Found the matrix (isn't bmat a sparse matrix? I'm getting an error telling
    #me that I'm not using sparse matrices but I thought that's what that was...)
    A = sparse.coo_matrix(sparse.bmat(x))
    
    #This block is for testing purposes
    """ comment this line to make code run
    plt.spy(A, markersize = 1)
    plt.show()
#    """
    
    return A
    raise NotImplementedError("Problem 5 Incomplete")

# Problem 6
def prob6(n = 10):
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    timeStart = 0
    timeEnd = 0
    
    csr = []
    notCsr = []
    x = []
    
    for i in range(1, n):
        A = prob5(i)
        Acsr = A.tocsr()
        
        #Timing the events
        timeStart = time.clock()
        myVector = np.random.rand(i**2, 1)
        ret = sparse.linalg.spsolve(Acsr, myVector)
        timeEnd = time.clock()
        
        csr.append(timeEnd - timeStart)
        
        timeStart = time.clock()
        A = Acsr.toarray()
        ret2 = la.solve(A, myVector)
        timeEnd = time.clock()
        notCsr.append(timeEnd - timeStart)
        
        x.append(i**2)
    
    #plotting the solve times
    myPlt = plt.subplot(121)
    myPlt.loglog(x, csr, basex = 2, basey = 2, label = "CSR solve")
    myPlt.loglog(x, notCsr, basex = 2, basey = 2, label = "Array solve")
    
    plt.title("Plots of time for solving (log base 2)")
    plt.legend()
    
    myPlt = plt.subplot(122)
    myPlt.plot(x, csr, label = "CSR solve")
    myPlt.plot(x, notCsr, label = "Array solve")
    
    plt.title("Plots of time for solving")
    plt.legend()
    
    return
    raise NotImplementedError("Problem 6 Incomplete")
