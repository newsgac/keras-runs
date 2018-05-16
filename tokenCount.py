#!/usr/bin/python3 -W all
"""
    tokenCount.py: count the average number of chars, tokens and sents per label
    usage: tokenCount.py < file
    20180516 erikt(at)xs4all.nl
"""

import re
import sys

def getLabel(line):
    tokens = line.split()
    label = tokens.pop(0)
    line = " ".join(tokens)
    return(label,line)

def countChars(line):
    return(len(line))

def countTokens(line):
    tokens = line.split()
    return(len(tokens))

def countSents(line):
    tokens = line.split()
    nbrOfSents = 0
    for i in range(0,len(tokens)):
        if re.search(r"^[.!?]+$",tokens[i]) or i == len(tokens)-1: 
            nbrOfSents += 1
    return(nbrOfSents)

def main(argv):
    nbrOfChars = {}
    nbrOfTokens = {}
    nbrOfSents = {}
    nbrOfLabels = {}
    for line in sys.stdin:
        label, line = getLabel(line)
        if not label in nbrOfLabels:
            nbrOfChars[label] = 0
            nbrOfTokens[label] = 0
            nbrOfSents[label] = 0
            nbrOfLabels[label] = 0
        nbrOfChars[label] += countChars(line)
        nbrOfTokens[label] += countTokens(line)
        nbrOfSents[label] += countSents(line)
        nbrOfLabels[label] += 1
    for label in nbrOfLabels:
        print(label,str(int(nbrOfSents[label]/nbrOfLabels[label])), \
                    str(int(nbrOfTokens[label]/nbrOfLabels[label])), \
                    str(int(nbrOfChars[label]/nbrOfLabels[label])))

if __name__ == "__main__":
    sys.exit(main(sys.argv))
