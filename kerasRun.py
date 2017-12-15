#!/usr/bin/python3
"""
    kerasRun.py: apply keras to newsgac data
    usage: kerasRun.py < file
    note: input line format: label token1 token2 ...
    source: https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py
    20171215 erikt(at)xs4all.nl
"""

import sys
import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer

COMMAND = sys.argv[0]
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
                lastElement += 1
                myDict[listIn[i]] = lastElement
            listOut.append(myDict[listIn[i]])
    return(listOut)

def readData():
    text = []
    classes = []
    for line in sys.stdin:
        fields = line.split()
        c = fields.pop(0)
        text.append(fields)
        classes.append(c)
    return({"text":makeNumeric(text), "classes":makeNumeric(classes)})

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
    return(score[1])

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
    
def main(argv):
    readDataResults = readData()
    text = readDataResults["text"]
    classes = readDataResults["classes"]
    averageScore = run10cv(text,classes)
    print("Average: ",averageScore)
    sys.exit(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
