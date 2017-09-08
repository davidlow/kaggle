import numpy as np

class ID3:

    def __init__(self, dataset, attributes_for_splitting, target_attribute):
        self.dataset = dataset
        self.t_attr  = target_attribute
        self.attr    = attributes_for_splitting

    @staticmethod
    def summarize(dataset, attr):
        '''
        dataset: list of dictionaries, each element is a different point,
                 dictionary keys are the attributes
        attr:    list of attributes that I want to sumarize.  

        Returns a dictionary where each key is an attribute A in attr and
        the value is a dictionary.  In that dictionary, each key is a 
        value V that the data had for attribute A and the value is the 
        number of datapoints with attribute A having value V.
        '''
        summary = {}
        for a in attr:
            summary[a] = {}
        for d in dataset:
            for a in attr:
                try:
                    summary[a][d[a]] += 1
                except:
                    summary[a][d[a]] = 1
        return summary

    @staticmethod
    def entropy(dataset, t_attr):
        '''
        H = \sum_{x \in X} p(x) \log2{p(x)}

        where: 
        - X are sets where all datapoints have the same value for t_attr,
        - p(x) is the ratio of the number of datapoints with t_attr = x 
        and the total number of datapoints

        dataset: list of dictionaries, each element is a different point,
                 dictionary keys are the attributes
        t_attr: single attribute
                target attribute that I want to find the entropy with 
                 respect to 
        '''
        summary = ID3.summarize(dataset, [t_attr])[t_attr]
        H = 0
        total = sum([s for s in summary.values()])
        for v in summary.values():
            H += v/total * np.log2(v/total)

        return H
    
    @staticmethod
    def split(dataset, attr):
        '''
        Splits dataset with respect to attribute attr

        Returns a dictionary where the keys are the different values
        each datapoint could have for attribute attr.  The values are 
        every data point that has that key
        '''
        # summary is a dictionary where each key is one of the possible
        # values that the data might take for attribute attr
        summary = ID3.summarize(dataset, [attr])[attr]

        keys = summary.keys()
        splitset = {}
        for k in keys:
            splitset[k] = []

        for d in dataset:
            splitset[d[attr]].append(d)

        return splitset
        

    @staticmethod
    def information_gain(dataset, attr, t_attr):
        '''
        IG = H(S) - \sum_{t\in T} p(t)H(t)

        where:
        - H(D) is the entropy of dataset D with respect to attribute
          t_attr
        - t is the set of data points where every point's attribute
          attr has the same value
        - T is a set of t's such that every point in S is in T exactly once

        dataset: list of dictionaries, each element is a different point,
                 dictionary keys are the attributes
        attr:    attribute to split over
        t_attr:  target attribute 
        '''
        IG = ID3.entropy(dataset, t_attr)

        splitsubset = ID3.split(dataset, attr)       

        for s_set in splitsubset.values():
            IG -= len(s_set)/len(dataset) * ID3.entropy(s_set, t_attr)

        return IG

    def id3(self, data, attr, willspliton):
        root = Node(willspliton = willspliton)

        targetsum = ID3.summarize(data, attr + [self.t_attr])[self.t_attr]

        # If all data has the same value for the target attribute
        # return a node with that value as the label
        for k in targetsum.keys():
            if targetsum[k] == len(data):
                root.label = k
                root.confidence=1
                root.data = data
                return root

        # If there are no more attributes to split over
        # return a node whose label is the value that the majority
        # of the data has for the target attribute
        if len(attr) == 0:
            keys = list(targetsum.keys())
            vals = list(targetsum.values())
            root.label = keys[np.argmax(vals)]
            root.confidence=targetsum[root.label]/len(data)
            root.data = data
            return root
        
        # Find the attribute that gives the largest information gain
        # and split the dataset into subsets where every point in the 
        # subset has the same value of that attribute
        igs     = [ID3.information_gain(data, a, self.t_attr) 
                    for a in attr]
        a_split = attr[np.argmax(igs)]
        splitset= ID3.split(data, a_split)

        root.spliton = a_split

        newattr = [a for a in attr if a != a_split]
        vals = list(splitset.values())
        keys = list(splitset.keys())
        for i in range(len(vals)):
            if len(vals[i]) == 0:
                print('How did this happen?')
            else:
                root.addchild(
                        self.id3( vals[i], 
                                  attr=newattr, 
                                  willspliton=root.willspliton + [(a_split,keys[i])]
                                )
                            )
                
        return root

# add how to navigate the tree
class Node(object):
    def __init__(self, willspliton='', label='', confidence=0, spliton=''):
        self._label = label
        self.children = []
        self.willspliton = willspliton
        self.spliton = spliton
        self.confidence = confidence

    def __repr__(self):
        s =  'Node({0}({4:2.2f}), willspliton={1}, spliton={2}, numchild={3})'.format(
                    self.label, 
                    self.willspliton, 
                    self.spliton, 
                    len(self.children),
                    self.confidence
                    )
        return s

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, l):
        self._label = l

    def addchild(self, node):
        self.children.append(node)

    def eval(self, datapt):
        if len(self.children) == 0:
            return self.label
        for c in self.children:
            if c.willspliton[-1][1] == datapt[self.spliton]:
                return c.eval(datapt)
        return -1

        





"""
def information_gain(dataset, a, t_a, ATTR):
    '''
    Calculate the information gain of a dataset
    '''
    
    
def entropy(dataset, a, ATTR):
    '''
    Calculate the entropy of dataset with respect to attribute a
    We calculate the entropy over the target attribute, the
    classifier.

    H(s) = \sum_{x\inX} p(x) \log2{p(x)}

    X is the set of classes 
    '''
    occur_of_attr = {}

    # For every entry in dataset
    for d in dataset:

        # If value of the key in question has not been seen before
        # Add it
        if d[ATTR[a]] not in occur_of_attr.keys():
            occur_of_attr[d[ATTR[a]]] = 0

        occur_of_attr[d[ATTR[a]]] += 1
    
    # I now have a dictionary where the keys are all the possible
    # values for ATTR[a] (the attribute we are interested in) in 
    # dataset.  The values are the number of times each key has 
    # occured in dataset.

    H = 0
    total = sum(occur_of_attr.values())
    for val in occur_of_attr.values():
        H += (val/total) * np.log2(val/total)
        
    return H

def id3(data, attr, t_attr, 
        ATTR,
        t_attr_val = [0,1]):
    '''
    data:       input data as dict
    attr:       array of attribute names
    t_attr:     index of target attribute, the one that decides
    ATTR:       static array of all attribute keys
    t_attr_val  the value that t_attr takes
    '''

    # If all datas' target attributes is one of t_attr_vals, 
    # return a node with label that value!
    for val in t_attr_val:
        allval = True
        for d in data:
            if d[ATTR[t_attr]] not val:
                allval = False
                break
        if allval:
            return Node(label=val)

    # If there are no other attributes to split over,
    # return a node whose label is the value is the majority
    if len(attr) == 0:
        majority = [0 for i in range(len(t_attr_val))]
        for d in data:
            for v in range(len(t_attr_val)):
                if d[ATTR[t_attr]] == t_attr_val[v]:
                    majority[v] += 1
        return Node(label=t_attr_val[np.argmax(majority)])

    # find the attribute we want to split on
    for a in attr:
        pass
"""
