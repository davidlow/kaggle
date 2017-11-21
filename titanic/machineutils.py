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

def eval_tree_sum(testset, root, target):
    [numcorrect, numwrong] = eval_tree(testset, root, target)
    return numcorrect/(numcorrect + numwrong)

def learnertoresult(learner, test):
    result = []
    for t in test:
        result.append( (t, learner(t)))
    return result

def cleanresult(result, boundary=.5, highval=1, lowval=0):
    for i in range(len(result)):
        if result[i][1] < boundary:
            result[i] = (result[i][0], lowval)
        else:
            result[i] = (result[i][0], highval)
    return result

def evallearner(learner, test, target, boundary=.5, highval=1, lowval=0):
    result = learnertoresult(learner, test)
    result = cleanresult(result, boundary, highval, lowval)
    numcorrect = 0
    for r in result:
        if r[0][target] == r[1]:
            numcorrect += 1
    return numcorrect / len(result)



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


#
#
#
#  BAGGING
#
#
#
#

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

#
#
#
#
# BOOSTING
#
#
#
#
#


def makeweightedlearner(train, mlalg, subsetsize, weights):

    # Generate a weak learner
    weightedset = []
    indexeschosen = []
    for i in range(subsetsize):
        rand = random.random()
        wsum = 0
        for i in range(len(weights)):
            if rand > wsum and rand < wsum + weights[i]:
                weightedset.append(train[i])
                indexeschosen.append(i)
                break
            wsum += weights[i]
    
    return [mlalg(weightedset), indexeschosen]

def adaboost(test, train, mlalg, subsetsize, T, target_attr, 
        valdict={1:1, 0:-1}):
    weights   = np.array([1.0/len(train) for t in train])

    errorfnct1 = lambda f, y, a: np.exp(-a*y*f)
    #errorfnct = lambda f, y, a: a*(f-y)**2

    train = np.array(train)
    test = np.array(test)
    trainval = np.array([valdict[t[target_attr]] for t in train])
    
    F0 = lambda x: 0
    F = F0
    corrections = []
    ws = []
    learners = []
    indexes = []
    ets = []
    ats = []
    numwrong = []

    for t in range(T):
        # create weighted learners
        ws.append(weights)
        [learner, index] = makeweightedlearner(train, 
                                      mlalg, 
                                      subsetsize, 
                                      weights)
        learners.append(learner)
        indexes.append(index)
        ht  = lambda x: learner.eval(x)
        htp = lambda x: valdict[learner.eval(x)]

        # Make error 
        #et = sum([weights[i] for i in range(len(train)) 
        #            if ht(train[i]) != train[i][target_attr]]
        #        )
        htptrain = np.array([htp(t) for t in train])
        et = np.sum(weights * (htptrain != trainval))

        # adaboost step size
        if et < 1e-30:
            et = 1e-30
        at = .5*np.log((1-et)/et)
        #print('[Error, at, numwrong] = [{0:2.2e},{1:2.2e},{2:2.2e}]'.format(
            #et, at, sum(htptrain != trainval)))

        # update learner
        # Fs.append(lambda x: Fs[t](x) + at*ht(x))
        # print(len(Fs))
        # Fs.append( ( lambda x: ( lambda y: Fs[x](y) + at*ht(y) ))(i) )
        # print(len(Fs))
        #Fs.append(lambda x: at*ht(x))
        F = lambda x: F(x) + at*ht(x)
        corrections.append(lambda x:at*ht(x))

        if et == 1e-30:
            break

        # update weights
        weights = np.multiply(
                    weights,
                    errorfnct1(htptrain, trainval, at)
                 )
        #print('[sumweights, maxweight, minweight] = [{0:2.2e},{1:2.2e},{2:2.2e}]'.format(
        #    sum(weights), max(weights), min(weights)))
        weights = weights/sum(weights)
        ats.append(at)
        ets.append(et)
        numwrong.append(sum(htptrain != trainval))

    F1 = lambda x: sum([ats[i] * learners[i].eval(x) for i in range(len(ats))])
    result = [(t, F1(t)) for t in test]

    return [F, F1, corrections, ws, result, learners, indexes, ats, ets, numwrong]

