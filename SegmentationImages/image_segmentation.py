# image_segmentation.py

from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian as lp
from scipy import linalg as la
from scipy.sparse.linalg import eigsh as eigsh
import scipy
import math

def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    n = len(A)
    D = np.zeros((n,n))
    for i in range(0, n):
        D[i][i] = sum(A[i])   
    L = D-A
    L2 = lp(A)
    if np.allclose(L2, L) is False:
        print("FAIL")
    return L


def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """

    #Get Laplacian
    B = laplacian(A)
    eig, eV = la.eig(B)
    eig = np.real(eig)
    eV = np.real(eV)
    eig = np.sort(eig)
    count = 0
    #Get count
    for i in range(0, len(eig)):
        if abs(eig[i]) < tol:
            eig[i] = 0
            count += 1

    return count, eig[1]


def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


class ImageSegmenter:
    """Class for storing and segmenting images."""

    def __init__(self, filename = "dream.png"):
        """Read the image file. Store its brightness values as a flat array."""
        
        #Reading image
        self.im = imread(filename)
        #Scaling image
        self.im = self.im/ 255
        #Storing info about image
        self.shape = self.im.shape
        self.min = self.im.min()
        self.max = self.im.max()
        self.t = self.im.dtype
        
        if len(self.shape) == 2:
            self.rgb = np.ravel(self.im)
        else:
            self.rgb = np.ravel(self.im.mean(axis = 2))                
        
        return

    def show_original(self):
        """Display the original image."""
        if len(self.shape) == 2:
            plt.imshow(self.im, cmap = "gray")
        else:
            plt.imshow(self.im)
        plt.axis("off")
        #Just felt like I should show what I made
        plt.show()
        return

    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        #Get the dimensions we'll be using
        mn = self.shape[0]*self.shape[1]
        w = 0
        
        #Make my sparse matrix mn x mn
        A = scipy.sparse.lil_matrix((mn, mn))
        
        #Make an array with mn elements
        D = np.zeros(mn)
        for i in range(0, len(self.rgb)):
        
            #Get the local values
            V, R = get_neighbors(i, r, self.shape[0], self.shape[1])
            mySum = 0
            for j in range(0, len(R)):
            
                #Applying equation 5.3
                #|B(i)-B(j)|/(sigmaBsq)
                t = np.abs(self.rgb[i] - self.rgb[V[j]])/sigma_B2
                
                #||X(i) - X(j)||/sigmaXsq
                y = np.abs(R[j])/sigma_X2
                
                #negative of the sum to an exp
                w = np.exp(-(t+y))
                
                #insert element of matrix
                A[i,V[j]] = w
                
                #Sum my column
                mySum = mySum + w
            
            #Add Diagonal entry
            D[i] = mySum
        
        return A.tocsc(), D

    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = scipy.sparse.csgraph.laplacian(A)
        
        #Construct D**-1/2
        B = scipy.sparse.diags(np.sqrt(1/D)).tocsc()
        
        C = B@L@B
        
        #This STUPID thing is FUNCTIONING PROPERLY!!
        w, C = eigsh(C, which = "SM", k = 2)
        """
        V = C[:, 1]
        
        a = self.im.shape[0]
        b = self.im.shape[1]
        K = np.zeros((a, b))
        
        #Kept getting an error so I put it in a try
        #except block
        K = V.reshape(a, b)
        mask = K > 0
        
        #"Return the mask"
        """
        mask = np.reshape(C[:, 1] > 0 , (self.im.shape[0], self.im.shape[1]))
        return mask
        
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        
        #Plt subplots
        #Get my Masks
        A, D = self.adjacency()
        mask = self.cut(A, D)
        nMask = ~mask
        
        a = plt.subplot(221)
        
        if len(self.shape) == 2:
            a.imshow(self.im, cmap = "gray")
        else:
            a.imshow(self.im)
        plt.axis("off")
        #Just felt like I should show what I made
        
        a = plt.subplot(222)
        
        #Show the image first part
        if len(self.shape) == 2:            
            oneIm = mask*self.im
            a.imshow(oneIm, cmap = "gray")
        else:            
            myMask = np.dstack((mask,mask,mask))
            oneIm = myMask*self.im
            a.imshow(oneIm)
        
        plt.axis("off")
        
        a = plt.subplot(223)
        
        #Show the image second part
        if len(self.shape) == 2:
            twoIm = (nMask)*self.im
            a.imshow(twoIm, cmap = "gray")
        else:
            nMyMask = np.dstack((nMask,nMask,nMask))
            twoIm = (nMyMask)*self.im
            a.imshow(twoIm)
        
        plt.axis("off")
        
        return
