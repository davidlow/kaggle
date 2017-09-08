import random

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
            print(root.eval(d), d[target])
            numwrong += 1
    return [numcorrect, numwrong]
