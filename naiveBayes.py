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
NBMODEL = MultinomialNB(alpha=0.1)

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

def showLabelNames(labelNames):
    ids = {}
    for label in labelNames:
        if labelNames[label] in ids: 
            sys.exit(COMMAND+": duplicate label id: "+labelNames[label])
        ids[int(labelNames[label])] = label
    for thisId in sorted(ids.keys()):
        print(str(thisId+1)+": "+ids[thisId])
    return()    

# From: https://stackoverflow.com/questions/11116697/
# how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    # top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    top = coefs_with_fns[:-(n + 1):-1]
    for (coef, fn) in top:
        print("\t%.4f\t%-15s\t" % (coef, fn))

def main(argv):
    try: dataFileName = sys.argv.pop(0)
    except: sys.exit(USAGE)
    data,labels = readData(dataFileName)
    dataN,countsModel,tfidfModel = makeNumericText(data)
    labelsN,labelNames = makeNumericList(labels)
    model = naiveBayesTrain(dataN,labelsN)
    show_most_informative_features(countsModel,model)
    sys.exit(0)
    accuracy,predictedLabels = naiveBayesTest(model,dataN,labelsN)
    showResult(labels,accuracy)
    accuracy,predictedLabels = naiveBayes10cv(dataN,labelsN,CV)
    showResult(labels,accuracy)
    showLabelNames(labelNames)
    print(metrics.confusion_matrix(labelsN,predictedLabels))
    sys.exit(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
