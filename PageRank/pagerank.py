# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""

import numpy as np
import csv
from scipy import linalg as la
import networkx as nx
import itertools


def prob1():
    
    A = np.array([[0,1,1,1], [0,0,0,0], [0,1,0,1], [0,0,1,0]])
    A = A.T
    
    x = DiGraph(A, labels = ["a","b","c","d"])
    
    print(x.labels)
    print(x.G)
    
    return x

def prob2():
    
    x = prob1()
    
    print("lin: ")
    p1 = x.linsolve()
    print(p1)
    
    print("eig: ")
    p2 = x.eigensolve()
    print(p2)
    
    print("iter: ")
    p3 = x.itersolve()
    print(p3)
    return x

def prob3():
    
    x = prob2()
    rank = get_ranks(x.linsolve())
    
    return rank
# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        #Change array from whatever into float
        A = A.astype(float)
        
        #Store the initial graph getting rid of sinks
        for i in range(0, len(A)):
            if np.sum(A[:, i]) == 0.:
                A[:, i] = np.ones(len(A))*1.0
            A[:,i] = A[:, i] / np.sum(A[:, i])
        self.G = A
        
        if labels == None:
            labels = list(np.linspace(0, len(A)-1, len(A), dtype = int))
        
        if len(labels) != len(A) or len(labels) != len(A[0]):
            raise ValueError("invalid label matrix combo")
            
        #Store the label
        self.labels = labels
        
        return 
        raise NotImplementedError("Problem 1 Incomplete")

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #get initial values
        A = np.eye(len(self.G)) - epsilon * self.G
        b = (1-epsilon) * np.ones(len(self.G))/len(self.G)
        
        #solve 
        p = la.solve(A, b)
        
        #return dictionary
        return dict(zip(self.labels, p))
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        
        #initialize
        E = np.ones(np.shape(self.G))
        B = epsilon * self.G + (1-epsilon) * E/len(self.G)
        
        #get eigs
        eig, vec = la.eig(B)
        
        #normalize
        p = vec[:, 0]/sum(vec[:, 0])
        
        #return dictionary
        return dict(zip(self.labels, p))
        
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #initialize
        i = 0
        pn1 = np.ones(len(self.G))/len(self.G)
        pn0 = np.zeros(len(self.G))
        
        #iterate
        while maxiter > i and la.norm(pn1-pn0, ord = 1) > tol:
            pn0 = pn1
            pn1 = epsilon * self.G @ pn0 + (1-epsilon)*np.ones(len(self.G))/len(self.G)
            i += 1
        
        #return the dictionary
        return dict(zip(self.labels, pn1))
        raise NotImplementedError("Problem 2 Incomplete")


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    
    #Sort the list by the keys of d, sorted by PageRank value from greatest to least.
    lsort = sorted(d, key = lambda k: d[k], reverse = True)
    
    return lsort
    
    raise NotImplementedError("Problem 3 Incomplete")


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon = 0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    with open(filename) as f:
        c = f.readlines()
    ids = []
    labels = []
    for i in c:
        k = i.split(sep = None)
        #Append labels and their connecting points
        ids.append(k[0].split("/"))
        labels.append(ids[-1][0])
        
    A = []
    
    #Make the matrix
    for i in range(0, len(labels)):
        vect = []
        for j in range(0, len(labels)):
            if labels[j] in ids[i][1:]:
                vect.append(1.)
            else:
                vect.append(0.)
        A.append(vect)
    

    #Solve
    A = np.array(A).T
    
    y = DiGraph(A, labels)
    
    x = y.itersolve(epsilon)
    
    return get_ranks(x)

    raise NotImplementedError("Problem 4 Incomplete")

def prob4(k = .85):
    return rank_websites(epsilon = k)[:3]
    

# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    wl = []
    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            wl.append(row)
    
    win = []
    lose = []
    
    #Get winners and losers...
    sorted(wl)
    labels = []
    for i in range(1, len(wl)):
        if wl[i][0] not in labels:
            labels.append(wl[i][0])
        if wl[i][1] not in labels:
            labels.append(wl[i][1])
        win.append(wl[i][0])
        lose.append(wl[i][1])
    
    A = np.zeros((len(labels), len(labels)))
    #Make the matrix
    for i in range(0, len(labels)):
        ind = []        
        count = lose.count(labels[i])
        if count > 0:
            ind = [lose.index(labels[i])]
        
        if count > 1:
            for k in range(0, count-1):
                ind.append(lose.index(labels[i], ind[-1] + 1))
        for j in range(0, len(ind)):
            winner = labels.index(win[ind[j]])
            A[i][winner] += 1.
            
    A = np.array(A).T
    y = DiGraph(A, labels)
    x = y.itersolve(epsilon)
    
    return get_ranks(x)
    raise NotImplementedError("Problem 5 Incomplete")

def prob5(ep):
    
    print(rank_ncaa_teams("ncaa2010.csv", epsilon = ep)[:9])
    
    return
# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    #Read in the actor information
    with open(filename, 'r', encoding = "utf-8") as f:
        c = f.readlines()
    actors = []
    labels = []
    for i in c:
        k = i.split("/")
        k[-1] = k[-1][:-1]
        actors.append(k[1:])
    
    #Make the graph
    g = nx.DiGraph()
    
    for i in range(0, len(actors)):
        for nodes in itertools.combinations(actors[i], 2):
            two, one = nodes[0], nodes[1]
            if g.has_edge(one, two):
                g[one][two]["weight"] += 1
            else:
                g.add_edge(one, two, weight = 1)
                
    x = nx.pagerank(g, alpha = epsilon)

    return get_ranks(x)
    
def prob6():
    return rank_actors(epsilon = .7)[:3]
    raise NotImplementedError("Problem 6 Incomplete")