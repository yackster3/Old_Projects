# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.spatial import KDTree
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    #Get the shape of X
    m, n = np.shape(X)
    d = []
    Y = np.copy(X)
    #Find all distances from X -> z
    for i in range(0, m):
        D = Y[i] - z
        d.append(la.norm(D))
    
    #Return the smallest value
    return Y[d.index(min(d))], min(d)

    raise NotImplementedError("Problem 1 Incomplete")


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    
    def __init__(self, x):
        
        if type(x) is not np.ndarray:
            raise TypeError("Not an np.ndarray")
        
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None
    
# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        
        #Insertion if nothing is there yet
        if self.root is None:
            self.root = KDTNode(data)
            self.k = len(data)
            self.root.pivot = 0
            return
        
        else:
            #Checks that the data has the same dinensions as other values in the tree
            if len(data) is not self.k:
                raise ValueError("Invalid length")
            
            #reucursive portion of insertion function
            def insertHelp(data, nextStep):
                
                """
                So what happens when these are equivalent?
                So this is just based off of the find function, but
                the instructions were not clear on how equivelance of 
                the checked element when the rest are not equivelant
                should be handled, it should work and I'm glad it did.
                
                Take for example x = [1,2,3,4,5] and y = [5,4,3,2,1] where 3 = 3
                but clearly x =/= y I figuered if I could find it then it would
                work, so I copied the code from find, after I checked what should 
                happen in this case, and asked a TA about it. I hope this is
                clearer as to what that single line meant before.
                """
                if nextStep.value[nextStep.pivot] > data[nextStep.pivot]:
                    
                    if nextStep.left is None:
                        #Insert Data on the left of this node
                        toInsert = KDTNode(data)
                        nextStep.left = toInsert
                        #Checks the correct element within the data for greater than/less than
                        toInsert.pivot = (nextStep.pivot + 1)%self.k
                        return
                    else:
                        #Continues searching the tree
                        insertHelp(data, nextStep.left)
                
                else:
                    #This checks that the inserted node is not equal to the node 
                    #we are currently checking. If it is it raises a value error
                    if np.allclose(data, nextStep.value):
                        raise(ValueError("Node already exists in the tree"))
                        
                    #same as insert left, but this time on the right
                    if nextStep.right is None:
                        #Insert something
                        toInsert = KDTNode(data)
                        nextStep.right = toInsert
                        toInsert.pivot = (nextStep.pivot + 1) % self.k
                        return
                    
                    else:    
                        insertHelp(data, nextStep.right)
                
                return
        
            #This calls the previous function to insert the node
            #through a recursive manner
            insertHelp(data, self.root)
            
        return
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        
        return self.NearestNeighborSearch(z)
        
        raise NotImplementedError("Problem 4 Incomplete")
    
    def NearestNeighborSearch(self, z):
        root = self.root
        
        def KDSearch(current, nearest, d):
            #Base case: dead end.
            if current is None:
                return nearest, d
        
            x = current.value
            i = current.pivot
            #Check if current is closer to z than nearest
            if la.norm(x - z) < d:
                nearest = current
                d = la.norm(x-z)
            #Search to the left
            if z[i] < x[i]:
                nearest, d = KDSearch(current.left, nearest, d)
                #Search to the right if needed
                if z[i] + d >= x[i]:
                    nearest, d = KDSearch(current.right, nearest, d)
            #Search to the right
            else:
                nearest, d = KDSearch(current.right, nearest, d)
                #Search to the left if needed
                if z[i] - d <= x[i]:
                    nearest, d = KDSearch(current.left, nearest, d)
            
            #Returning values
            return nearest, d
        
        node, d = KDSearch(root, root, la.norm(root.value - z))
        #Returning final answer
        return node.value, d
    
    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        #Check the shape of X matrix
        m, n = np.shape(X)
        if m != len(y):
            raise ValueError("X and Y dimensions don't match")
        #input the tree, store it's labels, and dimension
        self.kdTree = KDTree(X)
        self.label = y
        self.dim = len(X[0])
        
    def predict(self, z):
        #Checks that z could fit in the tree, and is thus comparable
        if len(z) == self.dim:
            #Gets my distances with their indices
            dist, index =  self.kdTree.query(z, k = self.n_neighbors)
            myLabels = []
            
            #so when n_neighbors is 1 we don't get an array
            #so the len(index) fails in the next step
            if self.n_neighbors == 1:
                index = [index]
            #Arranging the labels into one thing
            l = len(index)
            for i in range(0, l):
                myLabels.append(self.label[index[i]])
            
            #Gets the mode of my labels
            mod, cnt = stats.mode(myLabels)
            return mod
        else:
            raise ValueError("Invalid matrix dimension")

# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    
    data = np.load(filename)
    #Setting values
    X_train = data["X_train"].astype(np.float)
    Y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float)
    Y_test = data["y_test"]
    
    KN = KNeighborsClassifier(n_neighbors)
    KN.fit(X_train, Y_train)
    
    correct = 0
    wrong = 0
    
    #Compare answers
    for i in range(0, len(X_test)):
        
        if Y_test[i] == KN.predict(X_test[i]):
            correct += 1
        else:
#            plt.imshow(X_test[i].reshape((28,28)), cmap = "gray")
#            plt.show()
            wrong += 1
        
    #Give ratio of corret answers
    return correct / (correct + wrong)
    
    plt.imshow(X_test[0].reshape((28,28)), cmap = "gray")
    plt.show()
    
    return

    raise NotImplementedError("Problem 6 Incomplete")

def createTree(n = 100, k = 5):
    data = np.random.random((n,k))
    target = np.random.random(k)
    kdt = KDT()
    for i in range(0, n):
        try:
            kdt.insert(data[i])
        except ValueError:
            print("caught duplicate")
    return data, kdt, target