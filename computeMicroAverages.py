#!/usr/bin/python -W all
"""
    computeMicroAverages.py: compute micro averages of precision, recall, F1
    usage: computeMicroAverages.py < experiment.txt
    note: expected line format: label precision recall F1 gold_label_count
    20180522 erikt(at)xs4all.nl
"""

import sys

COMMAND = sys.argv.pop(0)

def main(argv):
    totalGoldLabelCount = 0
    totalCorrectCount = 0
    totalIdentifiedCount = 0
    for line in sys.stdin:
        fields = line.split()
        while fields[0] == "": fields.pop(0)
        try:
            label,precision,recall,f1,goldLabelCount = fields
        except:
            print(COMMAND+": unexpected input line: "+line)
        correctCount = round(float(goldLabelCount)*float(recall))
        if float(precision) <= 0: identifiedCount = 0
        else: identifiedCount = round(correctCount/float(precision))
        totalGoldLabelCount += round(float(goldLabelCount))
        totalCorrectCount += correctCount
        totalIdentifiedCount += identifiedCount
    if totalIdentifiedCount <= 0: averagePrecision = 0
    else: averagePrecision = totalCorrectCount/totalGoldLabelCount
    if totalGoldLabelCount <= 0: averageRecall = 0
    else: averageRecall = totalCorrectCount/totalGoldLabelCount
    if averagePrecision+averageRecall == 0: averageF1 = 0
    else: averageF1 = 2*averagePrecision*averageRecall/(averagePrecision+averageRecall)
    print("precision",averagePrecision,"recall",averageRecall,"F1",averageF1)
    if totalGoldLabelCount != totalIdentifiedCount:
        print(COMMAND+": warning: counting problem: "+str(totalGoldLabelCount)+" != "+str(totalIdentifiedCount))

if __name__ == "__main__":
    sys.exit(main(sys.argv))

