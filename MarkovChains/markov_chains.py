# markov_chains.py
"""Volume 2: Markov Chains.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        #Check that columns are stochastic
        for i in range(0, len(A)):
            x = 0
            for j in range(0, len(A)):
                x += A[j][i]
            if not np.isclose(x, 1):
#                print(i)
#                print(states[i])
                raise ValueError("Not column stochastic")
        
        self.mat = A
        
        #Add the labels
        if states is not None:        
            mySet = dict.fromkeys(states, set())
        else:
            states = []
            for j in range(0, len(A)):
                states.append(j)
            mySet = dict.fromkeys(states, set())
        
        #Finish dictionary
        for k in range(0, len(A)):
            mySet[states[k]] = set([k])
    
        self.dict = mySet
        return
        raise NotImplementedError("Problem 1 Incomplete")

    def __str__(self):
        s = str(self.dict) + "\n" + str(self.mat)
        return s
    
    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        #Find the label for the state
        label = self.dict[state]
        #For when I redfine the dictionary in the other part of the lab
        if type(label) is int:
#            print(self.mat[label])
#            print(state)
            result = np.random.multinomial(1, self.mat[:,label])
            index = np.argmax(result)
            for l, i in self.dict.items():
                if index == i:
                    return l
                
        #For when I don't need to redefine the dictionary
        for s in label:
            #Transition
            result = np.random.multinomial(1, self.mat[:,s])
            index = np.argmax(result)
            #Check where the transition took me
            for l, i in self.dict.items():
                if index in i:
                    return l
                
            
        raise NotImplementedError("Problem 2 Incomplete")

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        
        #initialize start
        labels = [start]
        for i in range(0, N-1):
            #transition
            labels.append(self.transition(labels[-1]))
        return labels
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        #initialize start
        labels = [start]
        while labels[-1] != stop:
            #transition
            labels.append(self.transition(labels[-1]))
        return labels
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        #Finds steady state
        x = np.random.rand(len(self.mat))
        x = x/sum(x)
        for i in range(0, maxiter):
            y = self.mat@x
            y = y/sum(y)
            #Returns steady state
            if np.allclose(x, y, atol = tol):
                return y
            x = y
        
        
        
        raise ValueError("Does not converge")
        
        raise NotImplementedError("Problem 4 Incomplete")

def MakeTransitionMatrix(filename = "trump.txt"):
    
    with open(filename, "r", encoding = "Latin-1") as f:
        content = f.readlines()
    
    vocab = set()
    #Get the vocabulary my bot has
    for i in range(0, len(content)):
        nextLine = content[i].split()
        for j in range(0, len(nextLine)):
            vocab.add(nextLine[j])
    
    vocab.add("$top")
    vocab.add("$tart")
    #Make my dictionary
    myDict = dict()
    index = 0
    for k in vocab:
        myDict.update({k : index})
        index += 1
        
#    print("Initializing matrix")
    #Initialize the matrix
    M = np.zeros((len(vocab), len(vocab)))
    for i in range(0, len(content)):
        
        nextLine = content[i].split()
        for j in range(-1, len(nextLine)+1):
            
            #Set current word
            if j == -1:
                curW = myDict["$tart"]
            elif j == len(nextLine):
                curW = myDict["$top"]
            else:
                curW = myDict[nextLine[j]]
            
            #Set next word
            if j+1 >= len(nextLine):
                nextW = myDict["$top"]
            else:
                nextW = myDict[nextLine[j+1]]
            
            M[nextW][curW] += 1
            
            
#    print("pre Stoch")
#    print(str(M[:,myDict["$tart"]]) + ", " + str(myDict["$tart"]))
    #Make M stochiactic
#    print(M)
    M = Stoch(M)
    
#    print("post Stoch")
#    print(str(M[:,myDict["$tart"]]) + ", " + str(myDict["$tart"]))
    
    return M, myDict

#    return M, list(vocab)

def Stoch(M):
    Mt = np.transpose(M)
    Mt = Mt/Mt.sum(axis=1)[:,None]
    
    return Mt.transpose()

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        
        What is up with this "Code can be improved for 
        SentenceGenerator.__init__()?" feedback? I mean, I didn't 
        have comments, but saying something as obscure as that is 
        like in problem 2 where you don't show the matrices labels
        for each column."
        """
        #Initialize a markov matrix to babble about
        M, l = MakeTransitionMatrix(filename)
        self.markov = MarkovChain(M)
        self.markov.mat = M
        self.markov.dict = l
        return
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        #Babble algorithm
        p = self.markov.path("$tart", "$top")
        output = ""
        for i in range(0, len(p)):
            if i > 0 and i < len(p)-1:
                output = output + p[i]
                if i != len(p) - 2:
                    output = output + " "
        #return the babbling string
        return output
        raise NotImplementedError("Problem 6 Incomplete")

def createMatrix():
    A = np.array([[.5,.3,.1,0],[.3,.3,.3,.3],[.2,.3,.4,.5],[0,.1,.2,.2]])
    
    B = np.array([[.7,.6],[.3,.4]])
    C = np.array([[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    c = np.array(["A","B","C","D"])
    return A, B, C,c