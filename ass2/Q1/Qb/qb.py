import sys
import os
import numpy as np
import random as random
import math
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from os.path import join
from wordcloud import WordCloud

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
    ans =[[0,0],[0,0]]
    for i in range(len(yTest)):
        if yTest[i]==1:
            if yPredic[i]==1:
                ans[0][0]+=1
            else:
                ans[1][0]+=1
        else:
            if yPredic[i]==1:
                ans[0][1]+=1
            else:
                ans[1][1]+=1
    return ans
def getPrecisionRecallF1(yPredic,yTest):
    conf =[[0,0],[0,0]]
    for i in range(len(yTest)):
        if yTest[i]==1:
            if yPredic[i]==1:
                conf[0][0]+=1
            else:
                conf[0][1]+=1
        else:
            if yPredic[i]==1:
                conf[1][0]+=1
            else:
                conf[1][1]+=1
    precision= conf[0][0]/(conf[0][0]+conf[0][1])
    recall= conf[0][0]/(conf[0][0]+conf[1][0])
    f1 = (2*(precision*recall))/(precision+recall)
    return precision,recall,f1

def main():
    global vocabulary,vocabularySize
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    out_dir = ""
    xTrain=[]
    yTrain=[]
    xTest=[]
    yTest=[]
    # Change the directory
    for file in os.listdir(join(train_dir,'pos')):
        if file.endswith(".txt"):
            file_path = join(join(train_dir,'pos'),file)
            curFile = open(file_path,'r')
            rfs = curFile.read().lower()
            xTrain.append(rfs.split())
            yTrain.append(1)
    # Change the directory
    for file in os.listdir(join(train_dir,'neg')):
        if file.endswith(".txt"):
            file_path = join(join(train_dir,'neg'),file)
            curFile = open(file_path,'r')
            rfs = curFile.read().lower()
            xTrain.append(rfs.split())
            yTrain.append(0)
    # Change the directory
    for file in os.listdir(join(test_dir,'pos')):
        if file.endswith(".txt"):
            file_path = join(join(test_dir,'pos'),file)
            curFile = open(file_path,'r')
            rfs = curFile.read().lower()
            xTest.append(rfs.split())
            yTest.append(1)
    # Change the directory
    for file in os.listdir(join(test_dir,'neg')):
        if file.endswith(".txt"):
            file_path = join(join(test_dir,'neg'),file)
            curFile = open(file_path,'r')
            rfs = curFile.read().lower()
            xTest.append(rfs.split())
            yTest.append(0)

    m = len(yTrain)
    testSize = len(xTest)
    ##part 2
    
    randomGuess = [(1 if random.random()>=0.5 else 0) for i in range(testSize)]
    allPosGuess=[1 for i in range(testSize)]
    print("for random generator accuracy is: ",getAcc(randomGuess,yTest))
    print("for all positive guess accuracy is: ",getAcc(allPosGuess,yTest))

    ##conf matrix save
    # print("for random guess:")
    file_path = join(out_dir,"conf_rand")
    curFile = open(file_path,'w')
    conf_rand=getConfMatrix(randomGuess,yTest)
    for i in range(2):
        curFile.write(str(conf_rand[i])+"\n")
    curFile.close()
    # print("for all positive guess:")
    file_path = join(out_dir,"conf_allpos")
    curFile = open(file_path,'w')
    conf_rand=getConfMatrix(allPosGuess,yTest)
    for i in range(2):
        curFile.write(str(conf_rand[i])+"\n")
    curFile.close()

main()