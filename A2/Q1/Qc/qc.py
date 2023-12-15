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

def main():
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    out_dir = "../Qa"
    ##conf matrix save
    print("for naive bayes:")
    file_path = join(out_dir,"conf_a")
    curFile = open(file_path,'r')
    print(curFile.readline(),curFile.readline())
    print("for random guess:")
    out_dir = "../Qb"
    file_path = join(out_dir,"conf_rand")
    curFile = open(file_path,'r')
    print(curFile.readline(),curFile.readline())
    print("for all positive guess:")
    file_path = join(out_dir,"conf_allpos")
    curFile = open(file_path,'r')
    print(curFile.readline(),curFile.readline())
main()