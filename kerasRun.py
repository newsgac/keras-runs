#!/usr/bin/python3
"""
    kerasRun.py: apply keras to newsgac data
    usage: kerasRun.py -T trainFile [ -t testFile ]
    note: input line format: label token1 token2 ...
    source: https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py
    20171215 erikt(at)xs4all.nl
"""

import getopt
import keras
import nltk
import numpy as np
import re
import sys
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold

COMMAND = sys.argv[0].split("/")[-1]
USAGE = "usage: "+COMMAND+" -T trainFile [ -t testFile ]"
RANDOMSTATE = 42
FOLDS = 9
CV = KFold(n_splits=FOLDS,shuffle=True,random_state=RANDOMSTATE)
MAXWORDS = 20000
BATCHSIZE = 196 # was 32
EPOCHS = 5
VERBOSE = 1
VALIDATIONSPLIT = 0.5
ANALYZER = "word"
MINDF = 0.0 # 0.01
MAXDF = 1.0 # 0.5
NGRAMMIN = 1
NGRAMMAX = 1
ALPHA=0.7156057222775337
HIDDENLAYER1 = 100
HIDDENLAYER2 = 200

def makeNumeric(listIn):
    myDict = {}
    listOut = []
    lastElement = -1
    for i in range(0,len(listIn)):
        if type(listIn[i]) is list:
            listOut.append([])
            for j in range(0,len(listIn[i])):
                if not listIn[i][j] in myDict:
                    lastElement += 1
                    myDict[listIn[i][j]] = lastElement
                listOut[i].append(myDict[listIn[i][j]])
        else:
            if not listIn[i] in myDict:
                if re.match("^__label__[0-9+]+$",listIn[i]):
                   nbr = re.sub("__label__","",listIn[i])
                   nbr = re.sub("\+.*$","",nbr)
                   nbr = int(nbr)
                   myDict[listIn[i]] = nbr
                   if nbr >= lastElement: lastElement = nbr+1
                elif re.match("^__label__None$",listIn[i]):
                   nbr = 0
                   myDict[listIn[i]] = nbr
                else:
                   lastElement += 1
                   myDict[listIn[i]] = lastElement
            listOut.append(myDict[listIn[i]])
    return(listOut,myDict)

def readData(inFileName):
    text = []
    classes = []
    try: inFile = open(inFileName,"r")
    except: sys.exit(COMMAND+": cannot read file "+inFileName)
    for line in inFile:
        fields = line.split()
        c = fields.pop(0)
        if len(fields) > 0 and re.search(r"DATE=",fields[0]):
            date = fields.pop(0)
        text.append(fields)
        classes.append(c)
    inFile.close()
    return({"text":text, "classes":classes})

# generates error message
def metricF1(yTrue,yPred):
    session = keras.backend.get_session()
    keras.backend.set_session(session)
    correct = 0.0
    yTrueValues = keras.backend.get_value(yTrue)
    yPredValues = keras.backend.get_value(yPred)
    for i in range(0,len(yPredValues)): 
        if yPredValues[i] == yTrueValues[i]: correct += 1.0
    return(keras.backend.variable(value=correct/len(yPred)))

def runExperiment(xTrain,yTrain,xTest,yTest,outFile):
    numClasses = np.max(yTrain) + 1
    tokenizer = Tokenizer(num_words=MAXWORDS)
    xTrain = tokenizer.sequences_to_matrix(xTrain, mode='binary')
    xTest = tokenizer.sequences_to_matrix(xTest, mode='binary')
    yTrain = keras.utils.to_categorical(yTrain, numClasses)
    yTest = keras.utils.to_categorical(yTest, numClasses)
    model = Sequential()
    model.add(Dense(HIDDENLAYER1, input_shape=(MAXWORDS,)))
    model.add(Activation('relu'))
    model.add(Dense(HIDDENLAYER2))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    history = model.fit(xTrain, yTrain,
                        batch_size=BATCHSIZE,
                        epochs=EPOCHS,
                        verbose=VERBOSE,
                        validation_split=VALIDATIONSPLIT)
    predictions = model.predict(xTest,batch_size=BATCHSIZE,verbose=VERBOSE)
    labelsN = []
    predictionsN = []
    for i in range(0,len(predictions)):
        maxJ = -1
        maxP = 0
        for j in range(0,len(predictions[i])):
            if predictions[i][j] > maxP:
                maxP = predictions[i][j]
                maxJ = j
        maxYJ = -1
        maxY = 0
        for j in range(0,len(yTest[i])):
            if yTest[i][j] > maxY:
                maxY = yTest[i][j]
                maxYJ = j
        labelsN.append(maxJ)
        predictionsN.append(maxYJ)
        print(maxYJ,maxJ,file=outFile)
    score = metrics.accuracy_score(labelsN,predictionsN)
    return(score,labelsN,predictionsN)

def singleRun(trainText,trainClasses,testText,testClasses,outFile):
    score, labelsN, predictionsN = runExperiment(np.array(trainText),np.array(trainClasses),np.array(testText),np.array(testClasses),outFile)
    return(score,labelsN,predictionsN)

def run10cv(text,classes,outFile):
    results = []
    labelsAll = []
    predictionsAll = []
    for n in range(0,FOLDS):
        testStart = int(float(n)*float(len(text))/float(FOLDS))
        testEnd = int(float(n+1)*float(len(text))/float(FOLDS))
        xTest = np.array(text[testStart:testEnd])
        yTest = np.array(classes[testStart:testEnd])
        xTrainList = text[:testStart]
        xTrainList.extend(text[testEnd:])
        xTrain = np.array(xTrainList)
        yTrainList = classes[:testStart]
        yTrainList.extend(classes[testEnd:])
        yTrain = np.array(yTrainList)
        score,labelsN,predictionsN = runExperiment(xTrain,yTrain,xTest,yTest,outFile)
        results.append(score)
        labelsAll.extend(labelsN)
        predictionsAll.extend(predictionsN)
        print("Fold: "+str(n)+"; Score: "+str(score),file=sys.stderr)
    total = 0.0
    for i in range(0,FOLDS): total += results[i]
    return(total/float(FOLDS),labelsAll,predictionsAll)

def processOpts(argv):
    argv.pop(0)
    try: options = getopt.getopt(argv,"T:t:",[])
    except: sys.exit(USAGE)
    trainFile = ""
    testFile = ""
    for option in options[0]:
        if option[0] == "-T": trainFile = option[1]
        elif option[0] == "-t": testFile = option[1]
    if trainFile == "": sys.exit(USAGE)
    return(trainFile,testFile)

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

def tokenizer(text):
    return(text.split())

def flatten(thisList):
    flatList = []
    for i in range(0,len(thisList)):
        thisMax = 0
        maxIndex = -1
        for j in range(0,len(thisList[i])):
            if thisList[i][j] > thisMax:
                maxIndex = j
                thisMax = thisList[i][j]
        flatList.append(maxIndex)
    return(flatList)

def sklearn10cv(text,labels):
    numClasses = np.max(labels) + 1
    model = Sequential()
    model.add(Dense(512, input_shape=(MAXWORDS,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    predictions = cross_val_predict(model,text,labels,cv=CV)
    return(metrics.accuracy_score(labels,predictedLabels),labels,predictions)

def showLabelNames(labelNames):
    ids = {}
    for label in labelNames:
        if labelNames[label] in ids:
            sys.exit(COMMAND+": duplicate label id: "+str(labelNames[label]))
        ids[int(labelNames[label])] = label
    for thisId in sorted(ids.keys()):
        print(str(thisId)+": "+ids[thisId],file=sys.stderr)
    return()

def makeOutFileName(fileName):
    return(fileName+"."+COMMAND+".out")

def main(argv):
    trainFileName, testFileName = processOpts(argv)
    trainData = readData(trainFileName)
    trainText = trainData["text"]
    trainClasses = trainData["classes"]
    outFileName = makeOutFileName(trainFileName)
    try: outFile = open(outFileName,"w")
    except: sys.exit(COMMAND+": cannot write file "+outFileName)
    if testFileName == "":
        trainText,myDict = makeNumeric(trainText)
        trainClasses,myDict = makeNumeric(trainClasses)
        showLabelNames(myDict)
        averageScore,labels,predictions = run10cv(trainText,trainClasses,outFile)
        print("Average: ",averageScore)
    else:
        testData = readData(testFileName)
        testText = testData["text"]
        testClasses = testData["classes"]
        combinedList = list(trainText)
        combinedList.extend(testText)
        numericData,myDict = makeNumeric(combinedList)
        testText = numericData[len(trainText):]
        trainText = numericData[:len(trainText)]
        combinedList = list(trainClasses)
        combinedList.extend(testClasses)
        numericData,myDict = makeNumeric(combinedList)
        showLabelNames(myDict)
        testClasses = numericData[len(trainClasses):]
        trainClasses = numericData[:len(trainClasses)]
        score,labels,predictions = singleRun(trainText,trainClasses,testText,testClasses,outFile)
        print("Score: ",score,file=sys.stderr)
    outFile.close()
    print(metrics.confusion_matrix(labels,predictions),file=sys.stderr)
    return(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
