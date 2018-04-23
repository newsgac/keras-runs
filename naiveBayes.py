#!/usr/bin/python3
# naiveBayes.py: apply naive bayes learning to genre data
# usage: naiveBayes.py dataFile
# 20180423 erikt(at)xs4all.nl

import numpy as np
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

COMMAND = sys.argv.pop(0)
CVSIZE = 10
USAGE = "usage: "+COMMAND+" data-file"
ANALYZER = "word"
NGRAMMIN = 1
NGRAMMAX = 5

def readData(dataFileName):
    data = []
    labels = []
    try: inFile = open(dataFileName,"r")
    except: sys.exit(COMMAND+": cannot read file "+dataFileName)
    for line in inFile:
        fields = line.split()
        if len(fields) >= 2:
            label = fields.pop(0)
            date = fields.pop(0)
            text = " ".join(fields)
            data.append(text)
            labels.append(label)
    inFile.close()
    return(data,labels)

def tokenizer(text):
    return(text.split())

def makeNumericText(texts):
    count_vect = CountVectorizer(
                 ngram_range=(NGRAMMIN,NGRAMMAX),
                 analyzer=ANALYZER,
                 tokenizer=tokenizer)
    text_counts = count_vect.fit_transform(texts)
    return(text_counts)

def makeNumericList(thisList):
    cellNames = {}
    thisListN = []
    seen = 0
    for i in range(0,len(thisList)):
        if not thisList[i] in cellNames:
            cellNames[thisList[i]] = seen
            seen += 1
        thisListN.append(cellNames[thisList[i]])
    return(thisListN,cellNames)

def naiveBayesTrain(examples,labels):
    return(MultinomialNB().fit(examples,labels))

def naiveBayesTest(model,examples):
    return(model.predict(examples))

def naiveBayes10cv(examples,labels,cvSize):
    return(cross_val_score(MultinomialNB(),examples,labels,cv=cvSize))

def compare(predictions,gold):
    correctCount = 0
    if len(predictions) != len(gold):
        sys.exit(COMMAND+": array length error in compare()\n")
    for i in range(0,len(predictions)):
        if predictions[i] == gold[i]: correctCount += 1
    return(correctCount)

def showResult(labels,correctCount):
    nbrOfLabels = len(labels)
    percentage = int(round(100*correctCount/len(labels)))
    print("correct:",str(correctCount),"of",str(nbrOfLabels),"("+str(percentage)+"%)")

def main(argv):
    try: dataFileName = sys.argv.pop(0)
    except: sys.exit(USAGE)
    data,labels = readData(dataFileName)
    dataN = makeNumericText(data)
    labelsN,labelNames = makeNumericList(labels)
    model = naiveBayesTrain(dataN,labelsN)
    predictions = naiveBayesTest(model,dataN)
    correctCount = compare(predictions,labelsN)
    showResult(labels,correctCount)
    scores = naiveBayes10cv(dataN,labelsN,CVSIZE)
    correctCount = int(round(np.mean(scores)*len(labels)))
    showResult(labels,correctCount)
    sys.exit(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
