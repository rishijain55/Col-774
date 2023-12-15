from cmath import inf
from math import gamma
import sys
import numpy as np

import pickle
import time
from os.path import join
from sklearn import svm



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
    for i in range(m):
        if data["labels"][i]==class1 and c1<lim:
            c1+=1
            y.append(-1.)
            x_temp=[]
            for j in range(32):
                for k in range(32):
                    x_temp.extend(data["data"][i][j][k])
            x.append(x_temp)
        elif data["labels"][i]==class2 and c2<lim:
            c2+=1
            y.append(1.)
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

def main():
    train_Dir = sys.argv[1]
    test_Dir = sys.argv[2]
    trainData = join(train_Dir,"train_data.pickle")
    testData = join(test_Dir,"test_data.pickle")
    x_train, y_train = get_Data(trainData,3,4)
    x_train=np.array(x_train)/255
    y_train=np.array(y_train)
    x_test, y_test = get_Data(testData,3,4)
    x_test=np.array(x_test)/255
    y_test=np.array(y_test)
    m = len(y_train)
    #Create a svm Classifier
    
    startTime = time.time()
    print("calculating for linear kernel")
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    timeTaken = time.time()- startTime
    print("number of support vectors is ",clf.support_.shape)
    sv_linear = clf.support_
    file_path = "support_vector_indices_sklearn_linear.txt"
    curFile = open(file_path,'w')
    
    for i in sv_linear:
        curFile.write(str(i)+" ")
    curFile.close()   
    print("w= ",clf.coef_)
    print("alphas= ",np.abs(clf.dual_coef_))
    print("b= ",clf.intercept_)
    print("accuracy by sklearn with linear kernel is:",getAcc(y_pred,y_test))
    print("time taken by sklearn with linear kernel is: ",timeTaken)
    
    
    ##gaussian
    startTime = time.time()
    print("calculating for gaussian kernel")
    clf = svm.SVC(kernel='rbf', gamma=0.001) # Linear Kernel
    #Train the model using the training sets
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    timeTaken = time.time()- startTime
    print("number of support vectors is ",clf.support_.shape)
    sv_linear = clf.support_
    file_path = "support_vector_indices_sklearn_gaussian.txt"
    curFile = open(file_path,'w')
    
    for i in sv_linear:
        curFile.write(str(i)+" ")
    curFile.close()  
    print("alphas= ",np.abs(clf.dual_coef_))
    print("b= ",clf.intercept_)
    print("accuracy by sklearn with gaussian kernel is:",getAcc(y_pred,y_test))
    print("time taken by sklearn with gaussian kernel is: ",timeTaken)
    


main()