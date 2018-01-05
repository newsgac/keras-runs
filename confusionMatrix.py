#!/usr/bin/python3
"""
    confusionMatrix.py: make confusion matrix
    usage: confusionMatrix.py < file
    note: expected line format: goldLabel WHITESPACE predictedLabel
    20180105 erikt(at)xs4all.nl
"""

import sys

COMMAND = sys.argv[0]
LABELS = ["NIE","REC","ACH","COL","OPI","VER","INT","HOO","REP"]
    
def main(argv):
    data = {}
    labels = {}
    for line in sys.stdin:
        fields = line.split()
        if len(fields) != 2: sys.exit(COMMAND+": unexpected line: "+line)
        gold,predicted = fields
        key = gold+" "+predicted
        labels[gold] = 1
        labels[predicted] = 1
        if not key in data: data[key] = 0
        data[key] += 1
    for gold in sorted(labels.keys()):
        print("{0:3s}".format(LABELS[int(gold)]),end="")
        for predicted in sorted(labels.keys()):
            key = gold+" "+predicted
            if not key in data: print("   .",end="")
            else: print("{0:4d}".format(data[key]),end="")
        print("")
    return(0)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
