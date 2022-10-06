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

print("best model is after trigram implementation without stemming:")
file_path = "metrics"
curFile = open(file_path,'r')
print(curFile.readline())
print(curFile.readline())
print(curFile.readline())
curFile.close()