import pandas as pd

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

print(dftrain.head())

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
print(dftrain.loc[0],y_train.loc[0])

