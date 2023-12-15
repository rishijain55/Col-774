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
from os.path import join
from matplotlib import pyplot as plt

def ker(x,z,gamma=0.001):
    XminY = x-z

    tunu =np.dot(XminY,XminY)
    return np.exp(-gamma*tunu)

def kernelMat(X,Y,gamma=0.001):
    K = np.zeros((X.shape[0],Y.shape[0]))*1.0
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            XminY = X[i]-Y[j]
            tunu =np.dot(XminY,XminY)
            K[i,j] = np.exp(-gamma*tunu)
    return K

def get_Data(dataFile, class1=3, class2=4):
    file = open(dataFile, 'rb')
    data = pickle.load(file)
    file.close()
    m = len(data["labels"])
    x=[]
    y=[]
    c1=0
    c2=0
    lim =inf
    x_noflat=[]
    for i in range(m):
        if data["labels"][i]==class1 and c1<lim:
            c1+=1
            y.append(-1.)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
            x_noflat.append(data["data"][i])
        elif data["labels"][i]==class2 and c2<lim:
            c2+=1
            y.append(1.)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
            x_noflat.append(data["data"][i])
    return x,y,x_noflat

def display_top5(indexSV,trainData):
    for i in range(5):
        imar = np.array(trainData[indexSV[i][1]])/255
        for c in range(32):
            for r in range(32):
                imar[c][r]=(imar[c][r][0],imar[c][r][1],imar[c][r][2])       
        # print(imar)
        fig = plt.figure()
        plt.imshow(imar, interpolation='none')
        fig.savefig('sv_gaussian{}.png'.format(i))
        plt.close(fig)

def getSvmParamsGaussian(x_train,y_train,C=1.0):
    gamma =0.001
    # x_train =x_train[0:10,0:5]
    # print(x_train)
    m,n = x_train.shape
    # tb = time.time()
    NormX = np.array([np.dot(x_train[i],x_train[i]) for i in range(m)])
    XXT = np.matmul(x_train,x_train.T)
    DifNorm= np.array([[NormX[i]+NormX[j]-2*XXT[i,j] for j in range(m)] for i in range(m)]) 
    K = np.exp(-gamma*DifNorm)
    # print("calc of k done")
    # print("time for calc is: ",time.time()-tb)
    # print(DifNorm)
    H = np.array([[(y_train[i]*y_train[j]*K[i,j])*1.0 for j in range(m)]for i in range(m)])
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix([[y_train[i]] for i in range(m)])
    b = cvxopt_matrix(np.zeros(1))

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    SVcount=0
    file_path = "support_vector_indices_cvx_gaussian.txt"
    curFile = open(file_path,'w')
    
    for i in range(m):
        if (alphas[i]>1e-5):
            SVcount+=1
            curFile.write(str(i)+" ")
    curFile.close()
    print("support vector count: ",SVcount,"total vectors: ",m)
    S = (np.logical_and((alphas > 1e-5),(C-1e-5>alphas))).flatten()
    x_valid =x_train[S]
    y_valid=y_train[S]
    b = y_valid[0] - np.sum(np.array([y_train[i]*alphas[i]*ker(x_train[i],x_valid[0]) for i in range(m)])) 
    # print("checking b")
    # print(b)
    #Display results
    indexSV = [(alphas[i],i) for i in range(m) if alphas[i] >1e-5]
    indexSV.sort(key = lambda x: x[0],reverse=True)
    print('Alphas = ',alphas[np.logical_and((alphas > 1e-5),(C-1e-5>alphas))])
    print('b = ', b)
    return alphas, b, indexSV

def testGaussianSVM(x_test,x_train,y_train,alphas,b):
    gamma =0.001
    m,n = x_train.shape

    testSize,_ = x_test.shape
    result =np.zeros(testSize)
    NormX_train = np.array([np.dot(x_train[i],x_train[i]) for i in range(m)])
    NormX_test = np.array([np.dot(x_test[i],x_test[i]) for i in range(testSize)])
    XXT = np.matmul(x_train,x_test.T)
    DifNorm= np.array([[NormX_train[i]+NormX_test[j]-2*XXT[i,j] for j in range(testSize)] for i in range(m)]) 
    K = np.exp(-gamma*DifNorm)
    for i in range(testSize):
        functionalMargin =np.sum(np.array([y_train[j]*alphas[j]*K[j,i] for j in range(m)])) +b
        # print(functionalMargin)
        if functionalMargin>=0.:
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
    testData = join(test_Dir,"test_data.pickle")
    x_train, y_train,x_train_noflat = get_Data(trainData,3,4)
    x_train=np.array(x_train)/255
    # print(x_train.shape)
    y_train=np.array(y_train)
    x_test, y_test,_ = get_Data(testData,3,4)
    x_test=np.array(x_test)/255
    y_test=np.array(y_test)
    m = len(y_train)
    startTime = time.time()
    print("calculating for gaussian kernel with cvxopt")
    alphas_gaussian,b_gaussian,indexSV=getSvmParamsGaussian(x_train,y_train)
    display_top5(indexSV,x_train_noflat)
    yPredicGaussian= testGaussianSVM(x_test,x_train,y_train,alphas_gaussian,b_gaussian)
    print("accuracy is ",getAcc(yPredicGaussian,y_test))
    timeTaken = time.time()-startTime
    print("time taken for gaussian kernel with cvxopt: ",timeTaken)


main()