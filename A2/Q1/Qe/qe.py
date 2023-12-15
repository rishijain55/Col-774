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
    
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    stemmedXTrain = [[ps.stem(w)  for w in xTrain[i]  ] for i in range(m)]
    # print("one done")
    stemmedXTest = [[ps.stem(w)  for w in xTest[i] ] for i in range(testSize)]
    # print("stemming done")
    filteredXTrain_big = [[w for w in stemmedXTrain[i]  if w not in stop_words] for i in range(m)]
    filteredXTest_big = [[w  for w in stemmedXTest[i]  if w not in stop_words] for i in range(testSize)]
    filteredXTrain_tig =filteredXTrain_big.copy() 
    filteredXTest_tig= filteredXTest_big.copy()
    xTrain_tig=xTrain.copy()
    xTest_tig=xTest.copy()

    
    for i in range(m):
        lfil = len(filteredXTrain_big[i])
        lx = len(xTrain[i])
        filteredXTrain_big[i].extend([filteredXTrain_big[i][j]+" "+filteredXTrain_big[i][j+1] for j in range(lfil-1)])
        xTrain[i].extend([xTrain[i][j]+" "+xTrain[i][j+1] for j in range(lx-1)])

    for i in range(testSize):
        lfil = len(filteredXTest_big[i])
        lx = len(xTest[i])
        filteredXTest_big[i].extend([filteredXTest_big[i][j]+" "+filteredXTest_big[i][j+1] for j in range(lfil-1)])
        xTest[i].extend([xTest[i][j]+" "+xTest[i][j+1] for j in range(lx-1)])


    thetaLogFilter_e_after_d_big,phiLogFilter_e_after_d_big = getNaiveBayesParams(filteredXTrain_big,yTrain)
    resultTestFilter_e_after_d_big = testNaiveBayes(filteredXTest_big,thetaLogFilter_e_after_d_big,phiLogFilter_e_after_d_big)
    resultTrainFilter_e_after_d_big = testNaiveBayes(filteredXTrain_big,thetaLogFilter_e_after_d_big,phiLogFilter_e_after_d_big)

    thetaLogFilter_e_after_a_big,phiLogFilter_e_after_a_big = getNaiveBayesParams(xTrain,yTrain)
    resultTestFilter_e_after_a_big = testNaiveBayes(xTest,thetaLogFilter_e_after_a_big,phiLogFilter_e_after_a_big)
    resultTrainFilter_e_after_a_big = testNaiveBayes(xTrain,thetaLogFilter_e_after_a_big,phiLogFilter_e_after_a_big)


    print("accuracy of naive bayes for bigram implementation after part a is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_a_big,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_a_big,yTest))

    print("accuracy of naive bayes for bigram implementation after part d is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_d_big,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_d_big,yTest))

    precision,recall,f1= getPrecisionRecallF1(resultTestFilter_e_after_d_big,yTest)
    print("precision is: ",precision)
    print("recall is: ",recall)
    print("F1-score is: ",f1)


    for i in range(m):
        lfil = len(filteredXTrain_tig[i])
        lx = len(xTrain_tig[i])
        filteredXTrain_tig[i].extend([filteredXTrain_tig[i][j-1]+" "+filteredXTrain_tig[i][j]+" "+filteredXTrain_tig[i][j+1] for j in range(1,lfil-1)])
        xTrain_tig[i].extend([xTrain_tig[i][j-1]+" "+xTrain_tig[i][j]+" "+xTrain_tig[i][j+1] for j in range(1,lx-1)])

    for i in range(1,testSize):
        lfil = len(filteredXTest_tig[i])
        lx = len(xTest_tig[i])
        filteredXTest_tig[i].extend([filteredXTest_tig[i][j-1]+" "+filteredXTest_tig[i][j]+" "+filteredXTest_tig[i][j+1] for j in range(1,lfil-1)])
        xTest_tig[i].extend([xTest_tig[i][j-1]+" "+xTest_tig[i][j]+" "+xTest_tig[i][j+1] for j in range(1,lx-1)])


    thetaLogFilter_e_after_d_tig,phiLogFilter_e_after_d_tig = getNaiveBayesParams(filteredXTrain_tig,yTrain)
    resultTestFilter_e_after_d_tig = testNaiveBayes(filteredXTest_tig,thetaLogFilter_e_after_d_tig,phiLogFilter_e_after_d_tig)
    resultTrainFilter_e_after_d_tig = testNaiveBayes(filteredXTrain_tig,thetaLogFilter_e_after_d_tig,phiLogFilter_e_after_d_tig)

    thetaLogFilter_e_after_a_tig,phiLogFilter_e_after_a_tig = getNaiveBayesParams(xTrain_tig,yTrain)
    resultTestFilter_e_after_a_tig = testNaiveBayes(xTest_tig,thetaLogFilter_e_after_a_tig,phiLogFilter_e_after_a_tig)
    resultTrainFilter_e_after_a_tig = testNaiveBayes(xTrain_tig,thetaLogFilter_e_after_a_tig,phiLogFilter_e_after_a_tig)


    print("accuracy of naive bayes for trigram implementation after part a is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_a_tig,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_a_tig,yTest))

    print("accuracy of naive bayes for trigram implementation after part d is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_d_tig,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_d_tig,yTest))

    precision,recall,f1= getPrecisionRecallF1(resultTestFilter_e_after_a_tig,yTest)
    print("precision is: ",precision)
    print("recall is: ",recall)
    print("F1-score is: ",f1)

    file_path = join(out_dir,"metrics")
    curFile = open(file_path,'w')
    curFile.write("precision is: "+str(precision)+"\n")
    curFile.write("accuracy is: "+str(recall)+"\n")
    curFile.write("F1-score is: "+str(f1)+"\n")
    curFile.close()
main()