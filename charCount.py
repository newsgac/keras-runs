#!/usr/bin/python3 -W all
"""
    charCount.py: compute the frequencies of characters, numbers and others
    usage: charCount.py < file
    note: outpus three frequency scores per line: chars, numbers, others
    20180504 erikt(at)xs4all.nl
"""

import re
import sys

def computeScores(line):
    chars, numbs, other, total = (0,0,0,0)
    for c in line:
        total += 1
        if re.search(r"[a-zA-Z]",c): chars += 1
        elif re.search(r"[0-9]",c): numbs += 1
        else: other += 1
    return(chars/total,numbs/total,other/total)

def printScores(myList):
    line = ""
    for i in myList:
        if len(line) > 0: line += " "
        line += str(i)
    print(line)

def main(argv):
    for line in sys.stdin:
        printScores(computeScores(line))

if __name__ == "__main__":
    sys.exit(main(sys.argv))
