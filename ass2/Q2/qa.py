import sys
import csv
import numpy as np
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
import pickle
import time
import os
from os.path import join

def get_Data(dataFile, class1=3, class2=4):
    file = open(dataFile, 'rb')
    data = pickle.load(file)
    file.close()
    m = len(data["labels"])
    x=[]
    y=[]
    for i in range(m):
        if data["labels"][i]==class1:
            y.append(-1.)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
        elif data["labels"][i]==class2:
            y.append(1.)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
    return x,y

def getSvmParams(x_train,y_train,C=1.0):
    m,n = x_train.shape
    xy_train=np.zeros((m,n))*1.
    for i in range(m):
        xy_train[i] = y_train[i] * x_train[i]
    H = np.matmul(xy_train,xy_train.T)
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    h = cvxopt_matrix([0.0] * m + [C] * m)
    A = cvxopt_matrix([[y_train[i]] for i in range(m)])
    b = cvxopt_matrix(np.zeros(1))

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    w=np.zeros(n)
    for i in range(m):
        w= w+(alphas[i]*xy_train[i])
    SVcount =0
    for i in range(m):
        if (alphas[i]>1e-5):
            SVcount+=1
    print("support vector count: ",SVcount,"total vectors: ",m)
    S = (np.logical_and((alphas > 1e-5),(C-alphas>1e-5))).flatten()
    b = y_train[S] - np.dot(x_train[S], w)
    print("checking b")
    print(b)
    #Display results
    print('Alphas = ',alphas[alphas > 1e-4])
    print('w = ', w.flatten())
    print('b = ', b[0])
    print(w.shape)
    return w, b[0]

def testSVM(x_test,w,b):
    m,n = x_test.shape
    result =np.zeros(m)
    for i in range(m):
        if (np.dot(w,x_test[i])+b)>=0.:
            result[i]=1
        else:
            result[i]=-1
    return result

def getAcc(yPredic,yTest):
    cor=0
    inc=0
    for i in range(len(yTest)):
        if yPredic[i]==yTest[i]:
            cor+=1
        else:
            inc+=1
    return 100*(cor/(cor+inc))

def main():
    train_Dir = sys.argv[1]
    test_Dir = sys.argv[2]
    trainData = join(train_Dir,"train_data.pickle")
    testData = join(train_Dir,"test_data.pickle")
    x_train, y_train = get_Data(trainData,0,1)
    x_train=np.array(x_train)/255
    y_train=np.array(y_train)
    x_test, y_test = get_Data(testData,0,1)
    x_test=np.array(x_test)/255
    y_test=np.array(y_test)
    m = len(y_train)
    w_linear,b_linear=getSvmParams(x_train,y_train)
    yPredicLinear= testSVM(x_test,w_linear,b_linear)
    print("accuracy is ",getAcc(yPredicLinear,y_test))


main()