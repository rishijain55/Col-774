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

def get_Data_train(dataFile):
    file = open(dataFile, 'rb')
    data = pickle.load(file)
    file.close()
    m = len(data["labels"])
    # print(m)
    # m =1000
    x=[]
    y=[[],[],[],[],[]]
    # print(y[0])
    c1=0
    c2=0
    lim =inf
    ind =0
    for i in range(m):
        # print(data["labels"][i][0])
        if(len(y[data["labels"][i][0]])<lim):
            y[data["labels"][i][0]].append(ind)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
            ind+=1
    return x,y
def get_Data_test(dataFile):
    file = open(dataFile, 'rb')
    data = pickle.load(file)
    file.close()
    m = len(data["labels"])
    # print(m)
    # m =1000
    x=[]
    y=[]
    # print(y[0])
    c=[0,0,0,0,0]
    lim =inf
    for i in range(m):
        # print(data["labels"][i][0])
        if(c[data["labels"][i][0]]<lim):
            c[data["labels"][i][0]]+=1
            y.append(data["labels"][i][0])
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
    return x,y

def getSvmParamsGaussian(x_train,y_train,C=1.0):
    gamma =0.001
    m,n = x_train.shape
    NormX = np.array([np.dot(x_train[i],x_train[i]) for i in range(m)])
    XXT = np.matmul(x_train,x_train.T)
    DifNorm= np.array([[NormX[i]+NormX[j]-2*XXT[i,j] for j in range(m)] for i in range(m)]) 
    K = np.exp(-gamma*DifNorm)
    # print("calc of k done")
    # print(DifNorm)
    H = np.array([[(y_train[i]*y_train[j]*K[i,j])*1.0 for j in range(m)]for i in range(m)])
    P = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((-1 * np.identity(m), np.identity(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix([[y_train[i]] for i in range(m)])
    b = cvxopt_matrix(np.zeros(1))

    #Run solver
    sol = cvxopt_solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    alphas = np.array(sol['x'])
    SVcount=0
    for i in range(m):
        if (alphas[i]>1e-5):
            SVcount+=1
    # print("support vector count: ",SVcount,"total vectors: ",m)
    S = (np.logical_and((alphas > 1e-5),(C-1e-5>alphas))).flatten()
    x_valid =x_train[S]
    y_valid=y_train[S]
    b = y_valid[0] - np.sum(np.array([y_train[i]*alphas[i]*ker(x_train[i],x_valid[0]) for i in range(m)])) 
    # print("checking b")
    # print(b)
    #Display results
    indexSV = [(alphas[i],i) for i in range(m) if alphas[i] >1e-5]

    # print('Alphas = ',alphas[np.logical_and((alphas > 1e-5),(C-1e-5>alphas))])
    # print('b = ', b)
    return alphas, b, indexSV

def testGaussianSVMforClass(x_test,x_train,y_train,alphas,b,c1,c2):
    print("testing for class ",c1,c2)
    gamma =0.001
    m,n = x_train.shape
    testSize,_ = x_test.shape
    result =[]
    NormX_train = np.array([np.dot(x_train[i],x_train[i]) for i in range(m)])
    NormX_test = np.array([np.dot(x_test[i],x_test[i]) for i in range(testSize)])
    XXT = np.matmul(x_train,x_test.T)
    DifNorm= np.array([[NormX_train[i]+NormX_test[j]-2*XXT[i,j] for j in range(testSize)] for i in range(m)]) 
    K = np.exp(-gamma*DifNorm)
    for i in range(testSize):
        functionalMargin =np.sum(np.array([y_train[j]*alphas[j]*K[j,i] for j in range(m)])) +b
        # print(functionalMargin)
        if functionalMargin>=0.:
            result.append((c2,functionalMargin))
        else:
            result.append((c1,abs(functionalMargin)))
    return result

def testSVM(x_test,x_train,y_train,params):
    testSize,_ = x_test.shape
    result =np.zeros(testSize)
    result_arr=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    for c1 in range(5):
        for c2 in range(c1+1,5):
            x_ob=[]
            y_ob=[]
            for i in y_train[c1]:
                x_ob.append(x_train[i])
                y_ob.append(-1.)
            for i in y_train[c2]:
                x_ob.append(x_train[i])
                y_ob.append(1.)
            result_arr[c1][c2]= testGaussianSVMforClass(x_test,np.array(x_ob),np.array(y_ob),params[c1][c2][0],params[c1][c2][1],c1,c2)
    # print(result_arr)
    for i in range(testSize):
        temp_count= [[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4]]
        for c1 in range(5):
            for c2 in range(c1+1,5):    
                vote =int(result_arr[c1][c2][i][0])
                conf = result_arr[c1][c2][i][1]
                # print(result_arr[c1][c2][i],kayo)
                temp_count[vote][0]+=1
                temp_count[vote][1]+=conf
        temp_count.sort(reverse=True)
        result[i]= temp_count[0][2]
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

def display_10(ind,trainData):
    file = open(trainData, 'rb')
    data = pickle.load(file)
    file.close()  
    for i in range(10):
        imar = np.array(data["data"][ind[i][0]])/255
        for c in range(32):
            for r in range(32):
                imar[c][r]=(imar[c][r][0],imar[c][r][1],imar[c][r][2])       
        # print(imar)
        fig = plt.figure()
        plt.imshow(imar, interpolation='none')
        fig.savefig('cvxopt_misclassed{}_as{}_no{}.png'.format(ind[i][2],ind[i][1],i))
        plt.close(fig)

def misclass(yPredic,yTest,testData):
    misInd=[]
    for i in range(len(yTest)):
        if yPredic[i]!=yTest[i]:
            misInd.append((i,yPredic[i],yTest[i]))
    random.shuffle(misInd)
    display_10(misInd[:10],testData)

def getConfMatrix(yPredic,yTest):
    ans =[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    for i in range(len(yTest)):
        yp = int(yPredic[i])
        yt = int(yTest[i])
        ans[yp][yt]+=1
    return ans

def trainSVMforClass(x_train,y_train,c1,c2):
    x_ob=[]
    y_ob=[]
    for i in y_train[c1]:
        x_ob.append(x_train[i])
        y_ob.append(-1.)
    for i in y_train[c2]:
        x_ob.append(x_train[i])
        y_ob.append(1.)
    print("training for class ",c1,c2)
    alphas_gaussian,b_gaussian,_=getSvmParamsGaussian(np.array(x_ob),np.array(y_ob))
    return alphas_gaussian,b_gaussian
    
def main():
    train_Dir = sys.argv[1]
    test_Dir = sys.argv[2]
    
    trainData = join(train_Dir,"train_data.pickle")
    testData = join(test_Dir,"test_data.pickle")
    x_train, y_train = get_Data_train(trainData)
    x_train=np.array(x_train)/255
    # print(x_train.shape)
    y_train=np.array(y_train)
    x_test, y_test = get_Data_test(testData)
    x_test=np.array(x_test)/255
    y_test=np.array(y_test)
    m = len(y_train)
    # print(y_train)

    startTime = time.time()
    print("calculating for gaussian kernel with cvxopt")
    params=[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    for i in range(5):
        for j in range(i+1,5):
            params[i][j]=trainSVMforClass(x_train,y_train,i,j)
    
    yPredicGaussian= testSVM(x_test,x_train,y_train,params)
    print("accuracy is ",getAcc(yPredicGaussian,y_test))
    timeTaken = time.time()-startTime
    misclass(yPredicGaussian,y_test,testData)
    file_path = "conf_a"
    curFile = open(file_path,'w')
    conf_cvx=getConfMatrix(yPredicGaussian,y_test)
    for i in range(5):
        curFile.write(str(conf_cvx[i])+"\n")
    curFile.close()
    print("time taken for gaussian kernel with cvxopt: ",timeTaken)


main()