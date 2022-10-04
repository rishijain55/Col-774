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
                ans[0][1]+=1
        else:
            if yPredic[i]==1:
                ans[1][0]+=1
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
    out_dir = "output"
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
    ##part 2
    
    randomGuess = [(1 if random.random()>=0.5 else 0) for i in range(testSize)]
    allPosGuess=[1 for i in range(testSize)]
    print("for random generator accuracy is: ",getAcc(randomGuess,yTrain))
    print("for all positive guess accuracy is: ",getAcc([1 for i in range(testSize)],yTrain))

    ##part 3
    print("confusion matrix are as follows")
    print("for naive bayes implementation")
    print(getConfMatrix(resultTest,yTest))
    print("for random guess:")
    print(getConfMatrix(randomGuess,yTest))
    print("for all positive guess:")
    print(getConfMatrix(allPosGuess,yTest))

    ##part 4
    
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()

    stemmedXTrain = [[ps.stem(w)  for w in xTrain[i]  ] for i in range(m)]
    print("one done")
    stemmedXTest = [[ps.stem(w)  for w in xTest[i] ] for i in range(testSize)]
    print("stemming done")
    filteredXTrain = [[w for w in stemmedXTrain[i]  if w not in stop_words] for i in range(m)]
    filteredXTest = [[w  for w in stemmedXTest[i]  if w not in stop_words] for i in range(testSize)]
    
    thetaLogFilter,phiLogFilter = getNaiveBayesParams(filteredXTrain,yTrain)
    resultTestFilter = testNaiveBayes(filteredXTest,thetaLogFilter,phiLogFilter)
    resultTrainFilter = testNaiveBayes(filteredXTrain,thetaLogFilter,phiLogFilter)

    print("accuracy of naive bayes after stemming is:")
    print("for training data: ",getAcc(resultTrainFilter,yTrain))
    print("for test data: ",getAcc(resultTestFilter,yTest))
    
    #word cloud
     #pos
    freq_lk_d,_= getFreq(filteredXTrain,yTrain)
    d = {}
    for a in vocabulary:
        d[a] = freq_lk_d[1][vocabulary[a]]
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    figdpos =plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    figdpos.savefig(join(out_dir, 'filtered_positive_class_word_cloud.png'))
    plt.close(figdpos)
     #neg
    d = {}
    for a in vocabulary:
        d[a] = freq_lk_d[0][vocabulary[a]]
    wordcloud = WordCloud()
    wordcloud.generate_from_frequencies(frequencies=d)
    figdneg =plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    figdneg.savefig(join(out_dir, 'filtered_negative_class_word_cloud.png'))
    plt.close(figdneg)
    
    ## part e
    for i in range(m):
        filteredXTrain[i].extend([filteredXTrain[i][j]+" "+filteredXTrain[i][j+1] for j in range(len(filteredXTrain[i])-1)])
        xTrain[i].extend([xTrain[i][j]+" "+xTrain[i][j+1] for j in range(len(xTrain[i])-1)])

    for i in range(m):
        filteredXTest[i].extend([filteredXTest[i][j]+" "+filteredXTest[i][j+1] for j in range(len(filteredXTest[i])-1)])
        xTest[i].extend([xTest[i][j]+" "+xTest[i][j+1] for j in range(len(xTest[i])-1)])


    thetaLogFilter_e_after_d,phiLogFilter_e_after_d = getNaiveBayesParams(filteredXTrain,yTrain)
    resultTestFilter_e_after_d = testNaiveBayes(filteredXTest,thetaLogFilter_e_after_d,phiLogFilter_e_after_d)
    resultTrainFilter_e_after_d = testNaiveBayes(filteredXTrain,thetaLogFilter_e_after_d,phiLogFilter_e_after_d)

    thetaLogFilter_e_after_a,phiLogFilter_e_after_a = getNaiveBayesParams(filteredXTrain,yTrain)
    resultTestFilter_e_after_a = testNaiveBayes(xTest,thetaLogFilter_e_after_a,phiLogFilter_e_after_a)
    resultTrainFilter_e_after_a = testNaiveBayes(xTrain,thetaLogFilter_e_after_a,phiLogFilter_e_after_a)


    print("accuracy of naive bayes for bigram implementation after part a is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_a,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_a,yTest))

    print("accuracy of naive bayes for bigram implementation after part d is:")
    print("for training data: ",getAcc(resultTrainFilter_e_after_d,yTrain))
    print("for test data: ",getAcc(resultTestFilter_e_after_d,yTest))

    ##part f
    precision,recall,f1= getPrecisionRecallF1(resultTestFilter_e_after_d,yTest)

    print("precision is: ",precision)
    print("accuracy is: ",recall)
    print("F1-score is: ",f1)
main()