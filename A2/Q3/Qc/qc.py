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
    print("for cvxopt:")
    file_path = join(out_dir,"conf_a")
    curFile = open(file_path,'r')
    print(curFile.readline(),curFile.readline(),curFile.readline(),curFile.readline(),curFile.readline())
    curFile.close()
    print("for cvxopt:")
    out_dir = "../Qb"
    file_path = join(out_dir,"conf_b")
    curFile = open(file_path,'r')
    print(curFile.readline(),curFile.readline(),curFile.readline(),curFile.readline(),curFile.readline())
    curFile.close()
main()