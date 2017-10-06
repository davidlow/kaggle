import random
from scipy import stats
import numpy as np

def splitdataset(dataset, ratio=.8):
    random.shuffle(dataset)
    train = []
    test  = []
    for d in dataset:
        if random.random() > ratio:
            test.append(d)
        else:
            train.append(d)

    return [train, test]

def eval_tree(testset, root, target):
    numcorrect = 0
    numwrong   = 0
    for d in testset:
        if root.eval(d) == d[target]:
            numcorrect += 1
        else:
            numwrong += 1
    return [numcorrect, numwrong]

def makeweaklearners(train, mlalg, subsetsize, subsetnum):
    '''
    '''
    subsets = []
    mlalgs  = []

    # Generate Di datasets by sampling train uniformly with replacements
    for i in range(subsetnum):
        di = [train[random.randint(0,len(train)-1)] 
                for j in range(subsetsize)]
        subsets.append(di)
    
    # For each dataset, create a weak learner associated with it
    for di in subsets:
        mlalgs.append(mlalg(di))

    # Return the weak learners
    return mlalgs


def bagging(test, train, mlalg, subsetsize, subsetnum, isclassifier=True):
    '''
    Returns the result from bagging

    inputs:
    test: (list) testint data set
    train: (list) training data set
    mlalg: (function) machine learning algorithm, takes in a training set
    subsetsize: (int) length of the bagging subsets 
    subsetnum: (int) number of subsets to bag over
    isclassifier: is the machine learning algorithm a classifier (mode) 
                 or a regression (mean)

    outputs:
    [result]
    result: (list of tuples), [(testpoint, value) ...]
    '''

    mlalgs = makeweaklearners(train, mlalg, subsetsize, subsetnum)

    result = []
    for testpoint in test:
        resultforalldi = np.array([ID3.eval(m, testpoint) for m in mlalgs])
        if isclassifier:
            result.append( (testpoint, stats.mode(resultforalldi)[0][0]))
        else:
            result.append( (testpoint, np.mean(resultforalldi)))
    return result

def eval_results(dataset, results, target_attr):
    numcorrect = 0
    numwrong   = 0
    for i in range(len(dataset)):
        if dataset[i][target_attr] == results[i][1]:
            numcorrect += 1
        else:
            print(dataset[i][target_attr], results[i][1])
            numwrong += 1
    return [numcorrect, numwrong]

def makeweightedlearner(train, mlalg, subsetsize, weights):

    # Generate a weak learner
    weightedset = []
    for i in range(subsetsize):
        rand = random.random()
        wsum = 0
        for i in range(len(weights)-1):
            if rand > wsum and rand < wsum + weights[i+1]:
                weightedset.append(train[i])
            wsum += weights[i]
    
    return mlalg(weightedset)

def adaboost(test, train, mlalg, subsetsize, T, target_attr, 
        valdict={1:1, 0:-1}):
    weights   = np.array([1.0/len(train) for t in train])
    errorfnct = lambda f, y: np.exp(-y*f)

    train = np.array(train)
    test = np.array(test)
    trainval = np.array([valdict[t[target_attr]] for t in train])
    
    F = lambda x: 0

    for t in range(T):
        # create weighted learners
        learner = makeweightedlearner(train, 
                                      mlalg, 
                                      subsetsize, 
                                      weights)
        ht  = lambda x: learner.eval(x)
        htp = lambda x: valdict[learner.eval(x)]

        # Make error 
        #et = sum([weights[i] for i in range(len(train)) 
        #            if ht(train[i]) != train[i][target_attr]]
        #        )
        htptrain = np.array([htp(t) for t in train])
        et = np.sum(weights * (htptrain != trainval))

        # adaboost step size
        at = .5*np.log((1-et)/et)

        # update learner
        F = lambda x: F(x) + at*ht(x)

        # update weights
        weights = np.multiply(
                    weights,
                    errorfnct(at*htptrain, trainval)
                 )
        weights = weights/sum(weights)
    
    return F

