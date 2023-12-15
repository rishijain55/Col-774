from cmath import inf
from math import gamma
import sys
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle
import time
from os.path import join
from sklearn import svm



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
        fig.savefig('misclass{}_as{}_no{}.png'.format(ind[i][2],ind[i][1],i))
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
    
    ##gaussian
    startTime = time.time()
    print("calculating for gaussian kernel")
    clf = svm.SVC(kernel='rbf', gamma=0.001,decision_function_shape='ovo') # Linear Kernel
    #Train the model using the training sets
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    timeTaken = time.time()- startTime
    print("accuracy by sklearn with gaussian kernel is:",getAcc(y_pred,y_test))
    file_path = "conf_b"
    curFile = open(file_path,'w')
    misclass(y_pred,y_test,testData)
    conf_cvx=getConfMatrix(y_pred,y_test)
    for i in range(5):
        curFile.write(str(conf_cvx[i])+"\n")
    curFile.close()
    print("time taken by sklearn with gaussian kernel is: ",timeTaken)
    


main()