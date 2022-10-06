from cmath import inf
from math import gamma
import sys
import csv
from matplotlib import test
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import pickle
import time
import os
from matplotlib import pyplot as plt
from os.path import join
import random 

def get_Data(dataFile):
    file = open(dataFile, 'rb')
    data = pickle.load(file)
    file.close()
    m = len(data["labels"])
    print(m)
    # m =1000
    x=[]
    y=[]
    # print(y[0])
    c=[0,0,0,0,0]
    lim =inf
    for i in range(0,10000,1000):
        fig = plt.figure()
        plt.imshow(np.array(data["data"][i]), interpolation='none')
        fig.savefig('asalme{}_no{}.png'.format(data["labels"][i],i))
        plt.close(fig)      
    return x,y

def main():
    train_Dir = sys.argv[1]
    test_Dir = sys.argv[2]
    
    trainData = join(train_Dir,"train_data.pickle")
    get_Data(trainData)

main()