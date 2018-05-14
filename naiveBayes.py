#!/usr/bin/python3
# naiveBayes.py: apply naive bayes learning to genre data
# usage: naiveBayes.py dataFile
# 20180423 erikt(at)xs4all.nl

import numpy as np
import sys
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn import metrics

COMMAND = sys.argv.pop(0)
RANDOMSTATE = 42
FOLDS = 10
CV = KFold(n_splits=FOLDS,shuffle=True,random_state=RANDOMSTATE)
USAGE = "usage: "+COMMAND+" data-file"
ANALYZER = "word"
MINDF = 0.01
MAXDF = 0.5
NGRAMMIN = 1
NGRAMMAX = 1
NBMODEL = BernoulliNB()

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
    countsModel = CountVectorizer(
                  analyzer=ANALYZER,
                  max_df=MAXDF,
                  min_df=MINDF,
                  ngram_range=(NGRAMMIN,NGRAMMAX),
                  tokenizer=tokenizer)
    textCounts = countsModel.fit_transform(texts)
    tfidfModel = TfidfTransformer()
    textTfidf = tfidfModel.fit_transform(textCounts)
    print(textTfidf.shape)
    return(textTfidf,countsModel,tfidfModel)

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
    return(NBMODEL.fit(examples,labels))

def naiveBayesTest(model,examples,labels):
    predictedLabels = model.predict(examples)
    return(metrics.accuracy_score(labels,predictedLabels),predictedLabels)

def naiveBayes10cv(examples,labels,cv):
    predictedLabels = cross_val_predict(NBMODEL,examples,labels,cv=cv)
    return(metrics.accuracy_score(labels,predictedLabels),predictedLabels)

def showResult(labels,accuracy):
    nbrOfLabels = len(labels)
    correctCount = int(round(nbrOfLabels*accuracy))
    percentage = int(round(accuracy*100))
    print("correct:",str(correctCount),"of",str(nbrOfLabels),"("+str(percentage)+"%)")

def main(argv):
    try: dataFileName = sys.argv.pop(0)
    except: sys.exit(USAGE)
    data,labels = readData(dataFileName)
    dataN,countsModel,tfidfModel = makeNumericText(data)
    labelsN,labelNames = makeNumericList(labels)
    model = naiveBayesTrain(dataN,labelsN)
    accuracy,predictedLabels = naiveBayesTest(model,dataN,labelsN)
    showResult(labels,accuracy)
    accuracy,predictedLabels = naiveBayes10cv(dataN,labelsN,CV)
    showResult(labels,accuracy)
    print(metrics.confusion_matrix(labelsN,predictedLabels))
    sys.exit(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
