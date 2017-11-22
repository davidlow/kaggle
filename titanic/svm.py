# Make a support vector machine framework


class svm(object):

    def __repr__(self):
        return 'svm()'

    def __init__(self, trainingset):
        self.trainingset = trainingset


    @staticmethod
    def maketrainingset(trainingdict, 
