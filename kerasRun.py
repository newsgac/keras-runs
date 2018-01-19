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
import numpy as np
import re
import sys
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

COMMAND = sys.argv[0]
USAGE = "usage: "+COMMAND+" -T trainFile [ -t testFile ]"
FOLDS = 10
MAXWORDS = 10000
BATCHSIZE = 32
EPOCHS = 5
VERBOSE = 0
VALIDATIONSPLIT = 0.1

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
                if re.match("__label__\d+",listIn[i]):
                   nbr = int(re.sub("__label__","",listIn[i]))
                   myDict[listIn[i]] = nbr
                   if nbr >= lastElement: lastElement = nbr+1
                elif re.match("__label__None",listIn[i]):
                   nbr = 0
                   myDict[listIn[i]] = nbr
                else:
                   lastElement += 1
                   myDict[listIn[i]] = lastElement
            listOut.append(myDict[listIn[i]])
    return(listOut)

def readData(inFileName):
    text = []
    classes = []
    try: inFile = open(inFileName,"r")
    except: sys.exit(COMMAND+": cannot read file "+inFileName)
    for line in inFile:
        fields = line.split()
        c = fields.pop(0)
        text.append(fields)
        classes.append(c)
    inFile.close()
    return({"text":text, "classes":classes})

def predict(xTest,yTest):
    predictions = model.predict(xTest,batch_size=BATCHSIZE,verbose=VERBOSE)
    for i in range(0,TESTSIZE):
        maxGold = 0.0
        maxGoldId = -1
        maxGuess = 0.0
        maxGuessId = -1
        for j in range(0,len(predictions[i])):
            if predictions[i][j] > maxGuess:
                maxGuess = predictions[i][j]
                maxGuessId = j
            if yTest[i][j] > maxGold:
                maxGold = yTest[i][j]
                maxGoldId = j
        print(str(maxGoldId)+" "+str(maxGuessId))

def runExperiment(xTrain,yTrain,xTest,yTest):
    numClasses = np.max(yTrain) + 1
    tokenizer = Tokenizer(num_words=MAXWORDS)
    xTrain = tokenizer.sequences_to_matrix(xTrain, mode='binary')
    xTest = tokenizer.sequences_to_matrix(xTest, mode='binary')
    yTrain = keras.utils.to_categorical(yTrain, numClasses)
    yTest = keras.utils.to_categorical(yTest, numClasses)
    model = Sequential()
    model.add(Dense(512, input_shape=(MAXWORDS,)))
    model.add(Activation('relu'))
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
    score = model.evaluate(xTest, yTest,
                           batch_size=BATCHSIZE, verbose=VERBOSE)
    p = model.predict(xTest,batch_size=BATCHSIZE,verbose=VERBOSE)
    for i in range(0,len(p)):
        maxJ = -1
        maxP = 0
        for j in range(0,len(p[i])):
            if p[i][j] > maxP:
                maxP = p[i][j]
                maxJ = j
        maxYJ = -1
        maxY = 0
        for j in range(0,len(yTest[i])):
            if yTest[i][j] > maxY:
                maxY = yTest[i][j]
                maxYJ = j
        print(str(maxYJ)+" "+str(maxJ))
    return(score[1])

def singleRun(trainText,trainClasses,testText,testClasses):
    score = runExperiment(np.array(trainText),np.array(trainClasses),np.array(testText),np.array(testClasses))
    return(score)

def run10cv(text,classes):
    results = []
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
        score = runExperiment(xTrain,yTrain,xTest,yTest)
        results.append(score)
        print("Fold: "+str(n)+"; Score: "+str(score))
    total = 0.0
    for i in range(0,FOLDS): total += results[i]
    return(total/float(FOLDS))

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

def main(argv):
    trainFile, testFile = processOpts(argv)
    trainData = readData(trainFile)
    trainText = trainData["text"]
    trainClasses = trainData["classes"]
    if testFile == "":
        trainText = makeNumeric(trainText)
        trainClasses = makeNumeric(trainClasses)
        averageScore = run10cv(trainText,trainClasses)
        print("Average: ",averageScore)
    else: 
        testData = readData(testFile)
        combinedList = list(trainText)
        combinedList.extend(testData["text"])
        numericData = makeNumeric(combinedList)
        testText = numericData[len(trainText):]
        trainText = numericData[:len(trainText)]
        combinedList = list(trainClasses)
        combinedList.extend(testData["classes"])
        numericData = makeNumeric(combinedList)
        testClasses = numericData[len(trainClasses):]
        trainClasses = numericData[:len(trainClasses)]
        score = singleRun(trainText,trainClasses,testText,testClasses)
        print("Score: ",score)
    return(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
