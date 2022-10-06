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

vocabulary={}
vocabularySize=0

def getFreq(xTrain,yTrain):
    global vocabularySize,vocabulary
    m =len(yTrain)
    freq_lk = np.zeros((2,vocabularySize))
    freq_k = np.zeros(2)

    for i in range(m):
        for word in xTrain[i]:
            freq_lk[yTrain[i]][vocabulary[word]]+=1
            freq_k[yTrain[i]]+=1
    return freq_lk,freq_k

def getNaiveBayesParams(xTrain,yTrain):
    global vocabularySize,vocabulary
    vocabulary={}
    vocabularySize=0
    m = len(yTrain)
    alpha=1
    phiLog =[0.0,0.]
    for i in range(m):
        phiLog[yTrain[i]]+=1.0
        for word in xTrain[i]:
            if word not in vocabulary:
                vocabulary[word]=vocabularySize
                vocabularySize+=1

    phiLog[0] = math.log2(phiLog[0])-math.log2(m)
    phiLog[1] = math.log2(phiLog[1])-math.log2(m)
    thetaLog = np.zeros((2,vocabularySize))
    freq_lk,freq_k=getFreq(xTrain,yTrain)
    for k in range(2):
        for l in range(vocabularySize):
            thetaLog[k][l]=math.log2((freq_lk[k][l]+alpha*1))- math.log2((freq_k[k]+alpha*vocabularySize))
    print(vocabularySize)
    return thetaLog,phiLog

def testNaiveBayes(xTest,thetaLog,phiLog):
    global vocabularySize,vocabulary
    ans=[]
    for x in xTest:
        p = -math.inf
        pos=0
        for k in range(2):
            p_k=phiLog[k]
            for word in x:
                if word in vocabulary:
                    p_k+= thetaLog[k][vocabulary[word]]
            if p_k>p:
                p = p_k
                pos =k
        ans.append(pos)
    return ans

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
    thetaLog,phiLog = getNaiveBayesParams(xTrain,yTrain)
    resultTest = testNaiveBayes(xTest,thetaLog,phiLog)
    resultTrain = testNaiveBayes(xTrain,thetaLog,phiLog)
    print("accuracy of naive bayes is:")
    print("for training data: ",getAcc(resultTrain,yTrain))
    print("for test data: ",getAcc(resultTest,yTest))

    #word cloud
     #pos
    freq_lk,_= getFreq(xTrain,yTrain)
    d = {}
    for a in vocabulary:
        d[a] = freq_lk[1][vocabulary[a]]
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    figapos =plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    figapos.savefig(join(out_dir, 'positive_class_word_cloud.png'))
    plt.close(figapos)
     #neg
    d = {}
    for a in vocabulary:
        d[a] = freq_lk[0][vocabulary[a]]
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    figaneg =plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    figaneg.savefig(join(out_dir, 'negative_class_word_cloud.png'))
    plt.close(figaneg)

    precision,recall,f1= getPrecisionRecallF1(resultTest,yTest)
    print("precision is: ",precision)
    print("recall is: ",recall)
    print("F1-score is: ",f1)

    file_path = join(out_dir,"conf_a")
    curFile = open(file_path,'w')
    conf_rand=getConfMatrix(resultTest,yTest)
    for i in range(2):
        curFile.write(str(conf_rand[i])+"\n")
    curFile.close()

main()