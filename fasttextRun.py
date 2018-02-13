#!/usr/bin/python3
"""
    fasttextRun.py: run fasttext via python api
    usage: fasttextRun.py -T trainFile -t testFile [-l]
    notes: input line format: label token1 token2 ...
           pyfasttext api: github.com/vrasneur/pyfasttext
    20180212 erikt(at)xs4all.nl
"""

from pyfasttext import FastText
import getopt
import os
import random
import re
import sys

COMMAND = sys.argv[0]
USAGE = "usage: "+COMMAND+" -T trainFile -t testFile [-l]"
DIM = 300
MINCOUNT = 5
NBROFFOLDS = 10
TMPFILENAME = COMMAND+"."+str(os.getpid())+"."+str(int(random.randrange(1000000)))
showLabels = False

def processOpts(argv):
    global showLabels

    argv.pop(0)
    try: options = getopt.getopt(argv,"T:t:l",[])
    except: sys.exit(USAGE)
    trainFile = ""
    testFile = ""
    for option in options[0]:
        if option[0] == "-T": trainFile = option[1]
        elif option[0] == "-t": testFile = option[1]
        elif option[0] == "-l": showLabels = True
    if trainFile == "": sys.exit(USAGE)
    return(trainFile,testFile)

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

def runExperiment(trainFileName,testFileName):
    global TMPFILENAME, DIM, MINCOUNT

    model = FastText()
    model.supervised(input=trainFileName,output=TMPFILENAME,dim=DIM,minCount=MINCOUNT,verbose=0)
    labels = model.predict_file(testFileName)
    data = readData(testFileName)
    correct = 0
    for i in range(0,len(labels)): 
        data["classes"][i] = re.sub("__label__","",data["classes"][i])
        if labels[i][0] == data["classes"][i]: correct += 1
    os.unlink(TMPFILENAME+".bin")
    os.unlink(TMPFILENAME+".vec")
    return({"correct":correct,"labels":labels})

def writeFile(fileName,text,labels):
    with open(fileName,"w") as f:
        for i in range(0,len(labels)):
            f.write(labels[i])
            for token in text[i]: f.write(" "+token)
            f.write("\n")
        f.close()

def printResults(correct,nbrOfLabels,prefix):
    print("{1:s}Correct: {0:0.1f}%".format(100*correct/nbrOfLabels,prefix))


def run10cv(trainFileName):
    global TMPFILENAME, NBROFFOLDS

    data = readData(trainFileName)
    classes = data["classes"]
    text = data["text"]
    trainFileName = TMPFILENAME+".train"
    testFileName = TMPFILENAME+".test"
    totalCorrect = 0
    labels = []
    for i in range(0,NBROFFOLDS):
        testStart = int(float(i)*float(len(text))/float(NBROFFOLDS))
        testEnd = int(float(i+1)*float(len(text))/float(NBROFFOLDS))
        writeFile(testFileName,text[testStart:testEnd],classes[testStart:testEnd])
        writeFile(trainFileName,text[:testStart]+text[testEnd:],classes[:testStart]+classes[testEnd:])
        results = runExperiment(trainFileName,testFileName)
        totalCorrect += results["correct"]
        labels.extend(results["labels"])
        if not showLabels:
            printResults(results["correct"],len(results["labels"]),"Fold: {0:2d}; ".format(i+1))
    os.unlink(trainFileName)
    os.unlink(testFileName)
    return({"correct":totalCorrect,"labels":labels})

def main(argv):
    global showLabels

    trainFileName, testFileName = processOpts(argv)
    if testFileName != "": results = runExperiment(trainFileName,testFileName)
    else: results = run10cv(trainFileName)
    if not showLabels: printResults(results["correct"],len(results["labels"]),"")
    else:
        for i in range(0,len(results["labels"])):
            print(results["labels"][i][0])
    return(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
