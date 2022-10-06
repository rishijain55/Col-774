from cmath import inf
from math import gamma
import sys
import numpy as np
import pickle
import time
from os.path import join
from sklearn import svm

import matplotlib.pyplot as plt

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

def getAcc(yPredic,yTest):
    cor=0
    inc=0
    for i in range(len(yTest)):
        if yPredic[i]==yTest[i]:
            cor+=1
        else:
            inc+=1
    return 100*(cor/(cor+inc))

def getConfMatrix(yPredic,yTest):
    ans =[[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    for i in range(len(yTest)):
        ans[yPredic[i]][yTest[i]]+=1
    return ans


def main():
    train_Dir = sys.argv[1]
    test_Dir = sys.argv[2]
    trainData = join(train_Dir,"train_data.pickle")
    testData = join(test_Dir,"test_data.pickle")
    x_train, y_train = get_Data(trainData)
    x_train=np.array(x_train)/255
    y_train=np.array(y_train)
    x_test, y_test = get_Data(testData)
    x_test=np.array(x_test)/255
    y_test=np.array(y_test)
    m = len(y_train)
    K_param=5
    ##get k fold data
    ind_array=np.random.permutation(m)
    x_train_per = np.copy(x_train)
    y_train_per = np.copy(y_train)

    for i in range(m):
        x_train_per[i]=x_train[ind_array[i]]
        y_train_per[i]=y_train[ind_array[i]]
    
    b = m//K_param
    print(b)
    x_train_kFold = np.array([x_train_per[i*b:i*b+b] for i in range(K_param)])
    y_train_kFold = np.array([y_train_per[i*b:i*b+b] for i in range(K_param)])
    # print(x_train_kFold)
    # print(y_train_kFold)

    C_poss = [1e-5, 1e-3, 1.0, 5.0, 10.0,50.0, 100.0]
    # C_poss = [50.0,100.0]
    accOnC_kFold=[0,0,0,0,0,0,0]
    accOnC_test=[0,0,0,0,0,0,0]
    for ind,C in enumerate(C_poss):
        current_accuracy = 0
        print('running on C =', C)
        for i in range(K_param):
            print('running on fold number', i + 1)
            # create a training set and the remaining set is a test set
            x_train_cur = np.concatenate(tuple([x_train_kFold[j] for j in range(K_param) if j != i]))
            y_train_cur = np.concatenate(tuple([y_train_kFold[j] for j in range(K_param) if j != i]))
            x_test_cur = x_train_kFold[i]
            y_test_cur = y_train_kFold[i]
            t = time.time()
            clf = svm.SVC(C=C,kernel='rbf', gamma=0.001,decision_function_shape='ovo')
            clf.fit(x_train_cur, y_train_cur)
            y_pred_cur = clf.predict(x_test_cur)
            accOnC_kFold[ind] += getAcc(y_pred_cur,y_test_cur)
            print('time taken:', time.time() - t)
        accOnC_kFold[ind] /= 5
        print(accOnC_kFold[ind])
        print('using C =', C, 'for testing on whole model')
        clf = svm.SVC(C=C,kernel='rbf', gamma=0.001,decision_function_shape='ovo')
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accOnC_test[ind]=getAcc(y_pred,y_test)
        print(accOnC_test[ind])

    xPlot = [np.log10(C_poss[i]) for i in range(len(C_poss))]
    y1Plot = accOnC_kFold
    y2Plot = accOnC_test
    fig = plt.figure()
    plt.plot(xPlot, y1Plot, label='kFold')
    plt.plot(xPlot, y2Plot, label='test')
    plt.xlabel('log10(C)')
    plt.savefig('qdplots.png')


main()