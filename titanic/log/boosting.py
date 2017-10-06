# IPython log file

# Fri, 22 Sep 2017 19:31:38
get_ipython().magic('run -i parser.py')
# Fri, 22 Sep 2017 19:32:14
dataset = parse('gender_submission.csv')
# Fri, 22 Sep 2017 19:32:24
get_ipython().magic('run -i machineutils.py')
# Fri, 22 Sep 2017 19:32:41
[train,test] = splitdataset(dataset)
