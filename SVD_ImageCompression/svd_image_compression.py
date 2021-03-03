# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    #Compute eigenvalues/vectors
    lam, V = la.eig((A.conj().T @ A))
    sig = np.sqrt(lam)
    
    #Sort results
    argB = np.argsort(sig)
    arg = []
    for i in range(0, len(argB)):
        arg.append(argB[len(argB)-1-i])
    sig = sig[arg]
    V = V[:,arg]
    #How many non-zero positive
    r = 0
    for j in range(0, len(sig)):
        if abs(sig[j]) >= tol:
            r += 1
    
    sig1 = sig[:r]
    V1 = np.array(V[:,:r])
    
#    print(np.shape(A))
#    print(np.shape(V1))
    U1 = A@V1
    U1 = U1/sig1
    
    #Return answers
    return U1, sig1, V1.conj().T

    raise NotImplementedError("Problem 1 Incomplete")

def test(A):
    U, S, V = compact_svd(A)
    S = np.diag(S)
    x = []
    if np.allclose(U.T@U, np.identity(len(A[0]))):
        x.append(True)
    else:
        x.append(False)
    if np.allclose(U@S@V, A):
        x.append(True)
    else:
        x.append(False)
    if np.linalg.matrix_rank(A) == len(S):
        x.append(True)
    else:
        x.append(False)
    return x

def tests(n = 100, k = 6, w = 5):
    a  = []
    b = []
    c = []
    for i in range(0, n):
        M = np.random.random((k,w))
        t = test(M)
        a.append(t[0])
        b.append(t[1])
        c.append(t[2])
    plt.hist(a)
    plt.show()
    plt.hist(b)
    plt.show()
    plt.hist(c)
    plt.show()
def unit(n = 200):
    l = np.linspace(0,2*np.pi, n)
    M = [np.cos(l),np.sin(l)]
    return M

def prob2(A = np.array([[1,3],[3,1]]), n = 100):
    visualize_svd(A, n)
    return
# Problem 2
def visualize_svd(A, n = 200):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    M = unit(n)
    E = np.array([[1,0],[0,0],[0,1]])
    E = E.T
    U, S, Vh = la.svd(A)
    S = np.diag(S)
    
    #No changes
    myPlt = plt.subplot(221)
    myPlt.plot(M[0], M[1], color = "green")
    myPlt.plot(E[0], E[1], color = "red")
    myPlt.axis("equal")
    #1 multiplication
    B = Vh@M
    E = Vh@E
    myPlt = plt.subplot(222)
    myPlt.plot(B[0], B[1], color = "green")
    myPlt.plot(E[0], E[1], color = "red")
    myPlt.axis("equal")
    #2 multiplications
    C = S@B
    E = S@E
    myPlt = plt.subplot(223)
    myPlt.plot(C[0], C[1], "green")
    myPlt.plot(E[0], E[1], color = "red")
    myPlt.axis("equal")
    #3 multiplication
    D = U@C
    E = U@E
    myPlt = plt.subplot(224)
    myPlt.plot(D[0],D[1], color = "green")
    myPlt.plot(E[0], E[1], color = "red")
    myPlt.axis("equal")
    return
    raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    U, S, V = la.svd(A)
    V = V.conj().T
    if s > len(S):
        raise ValueError( str(len(S)) + " = Rank(A) > s" )
    
    U2 = U[:,:s]
    S2 = S[:s]
    V2 = V[:,:s]
    V2 = V2.conj().T
    
    S2 = np.diag(S2)
    
    Ag = U2@S2@V2
    ent = U2.size + len(S2) + V2.size
    return Ag, ent
    
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    #Is this not just problem 1 repeated, just with different returns?
    
    u, d, v = compact_svd(A)
    
    s = len(d)
    for i in range(0, len(d)):
        if d[i] < err:
            s = i
            break
    
    v = v.conj().T
    U2 = u[:,:s]
    S2 = d[:s]
    V2 = v[:,:s]
    V2 = V2.conj().T
    
    S2 = np.diag(S2)
    
    ent = U2.size + len(S2) + V2.size
    
#    print(S2)
    return U2@S2@V2, ent
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def compress_image(filename = "hubble.jpg", s = 2):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    
    image = imread(filename)/255
    
    toShow = plt.subplot(121)
    
    #Color
    if len(np.shape(image)) == 3:
        #Set up original RGB values
        R = np.array(image[:,:,0])
        G = np.array(image[:,:,1])
        B = np.array(image[:,:,2])
        
        R, errR = svd_approx(R,s)
        G, errG = svd_approx(G,s)
        B, errB = svd_approx(B,s)
        imageF = np.dstack((R,G,B))
        err = errR + errG + errB
        toShow.imshow(imageF)
        toShow.set_title("New " + str(err) + ", so " + str((image.size-err)) + " \"saved\"")
        toShow.axis("off")
        
        toShow = plt.subplot(122)
        toShow.set_title("Original " + str(image.size))
        toShow = plt.imshow(image)
    #Gray
    else:
        imageF, err = svd_approx(image, s)
#        print(np.shape(imageF))
        toShow.imshow(imageF, cmap = "gray")
        toShow.set_title("New " + str(err) + ", so " + str((image.size-err)) + " \"saved\"")
        toShow.axis("off")
        toShow = plt.subplot(122)
        toShow.set_title("Original " + str(image.size))
        toShow = plt.imshow(image, cmap = "gray")
        
    plt.suptitle("MY PLOTS: " + str((image.size-err)) + " \"saved\" :p")
        
    return
    raise NotImplementedError("Problem 5 Incomplete")
