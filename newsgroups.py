#!/usr/bin/python3
# newsgroups.py: process the newsgroups data of sklearn
# usage: newsgroups.py
# 20180515 erikt(at)xs4all.nl

import nltk
from sklearn.datasets import fetch_20newsgroups

def tokenize(text):
    tokenizedSentenceList = nltk.word_tokenize(text)
    tokenizedText = " ".join(tokenizedSentenceList)
    return(tokenizedText)

results = fetch_20newsgroups(subset="train",
                             remove=('headers', 'footers', 'quotes'))
#for i in range(0,len(results['data'])):
#    print("__label__"+str(results["target"][i])+" "+tokenize(results["data"][i]))
for i in range(0,len(results["target_names"])):
    print(str(i)+": "+results["target_names"][i])
