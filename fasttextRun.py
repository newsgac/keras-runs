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
showLabels = False

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
    tmpFileName = COMMAND+"."+str(os.getpid())+"."+str(int(random.randrange(1000000)))
    model = FastText()
    model.supervised(input=trainFileName,output=tmpFileName,dim=DIM,minCount=MINCOUNT,verbose=0)
    labels = model.predict_file(testFileName)
    data = readData(testFileName)
    correct = 0
    for i in range(0,len(labels)): 
        data["classes"][i] = re.sub("__label__","",data["classes"][i])
        if labels[i][0] == data["classes"][i]: correct += 1
    correct *= 100/len(data["classes"])
    os.unlink(tmpFileName+".bin")
    os.unlink(tmpFileName+".vec")
    return({"correct":correct,"labels":labels})

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
    if trainFile == "" or testFile == "": sys.exit(USAGE)
    return(trainFile,testFile)

def main(argv):
    global showLabels

    trainFileName, testFileName = processOpts(argv)
    results = runExperiment(trainFileName,testFileName)
    if not showLabels: 
        print("correct: {0:0.1f}%".format(results["correct"]))
    else:
        for i in range(0,len(results["labels"])):
            print(results["labels"][i][0])
    return(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
