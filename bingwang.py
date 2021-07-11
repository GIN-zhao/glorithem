from libsvm.svmutil import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def svm_test():
    data = pd.read_csv('krkopt.csv')
    data = data.dropno(axis=0)
    print(data)


svm_test()
