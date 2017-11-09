# Perceptron

import numpy as np
import time


class Perceptron(object):


    def __repr__(self):
        return 'Perceptron()'

    def __str__(self):
        return 'Machine learning: Perceptron Algorithm'

    def __init__(self, trainingset, parameternames, targetname,
                targetconv = {0:-1, 1:1}):
        '''
        Create a Perceptron object.

        Parameters:
        trainingset: list of dictionaries, each entry is a datapt with
                    the keys being parameter names and the values being
                    the value of that parameter
        parameternames: list of strings of parameter names to target
        targetname: string of the name of the key parameter that determines
                    the value of the datapt
        targetconv: dictionary where the keys are possible choices for
                    datapt[targetname] and the values are the corrected
                    values
        '''
        self.train = trainingset
        self.pnames = parameternames
        self.tname  = targetname
        self.tconv  = targetconv

        self.makedataset()   

    def eval(self, datapt):
        return Perceptron.h(self.W, self.makedatapt(datapt))

    @staticmethod
    def h(W, datapt):
        return np.sign(np.dot(W, datapt))
    
    def makedatapt(self, datapt):
        if type(datapt) == np.ndarray:
            return datapt
        d = np.ones(len(self.pnames)+1)
        for n,i in zip(self.pnames, range(len(self.pnames))):
            d[i] = datapt[n]
        return d
    
    def makedataset(self):
        self.train_datapts = np.zeros(
                (len(self.train), len(self.pnames)+1))
        self.train_results = np.zeros(
                (len(self.train), 1))
        i = 0
        for pt in self.train:
            self.train_datapts[i] = self.makedatapt(pt)
            self.train_results[i] = self.tconv[pt[self.tname]]
            i += 1
    

    def makeW(self, timelimit=600):
        self.W = np.zeros(len(self.train_datapts[0]))
        starttime = time.time()
        while starttime + timelimit > time.time():
            m = 0
            for x,y in zip(self.train_datapts, self.train_results):
                if self.eval(x)*y <= 0:
                    self.W += y * x
                    m      += 1
                    print('  {0},{1}',x,y)
            print(m) 
            if m ==0:
                break
         
        return starttime + timelimit > time.time()

    @staticmethod 
    def accuracy(p, dataset):
        numcorrect = 0
        total = 0
        keys = p.tconv.keys()
        values = p.tconv.values()
        conv = {}
        for k,v in zip(keys,values):
            conv[v] = k
        for d in dataset:
            total += 1
            if d[p.tname] == conv[p.eval(d)]:
                numcorrect += 1
        print (numcorrect/total)
        return [numcorrect, total]

