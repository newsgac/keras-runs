#!/usr/bin/python3 -W all
"""
    cm2factors.py: extract re-estimation factors from confusion matrix
    usage: cm2factors.py < file.txt
    note: expects square array on input lines
    20180528 erikt(at)xs4all.nl
"""

import re
import sys

COMMAND = sys.argv.pop(0)

def cleanup(line):
    line = re.sub(r"[\[\]]","",line)
    line = re.sub(r"^\s+","",line)
    line = re.sub(r"\s+$","",line)
    line = re.sub(r"\s+"," ",line)
    return(line)

def computePrecision(array,index):
    total = 0
    correct = 0
    for i in range(0,len(array)):
        total += int(array[i][index])
        if i == index: correct += int(array[i][index])
    if total == 0: return(0)
    else: return(correct/total)

def computeRecall(array,index):
    total = 0
    correct = 0
    for i in range(0,len(array)):
        total += int(array[index][i])
        if i == index: correct += int(array[index][i])
    if total == 0: return(0)
    else: return(correct/total)

def computeFactors(array):
    for i in range(0,len(array)):
        precision = computePrecision(array,i)
        recall = computeRecall(array,i)
        print("%d : %0.2f %0.2f %0.2f" % (i,round(precision,2),round(recall,2),round(precision/recall,2)))

def main(argv):
    array = []
    for line in sys.stdin:
        line = cleanup(line)
        row = line.split()
        array.append(row)
        if len(array) == len(array[0]):
            computeFactors(array)
            array = []
    if len(array) > 0: 
        sys.exit(COMMAND+": unexpected array: "+str(array))

if __name__ == "__main__":
    sys.exit(main(sys.argv))

