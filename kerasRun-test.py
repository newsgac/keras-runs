#!/usr/bin/python3
"""
    kerasRun-test.py: tests for kerasRun.py
    usage: kerasRun-test.py
    20171216 erikt(at)xs4all.nl
"""

import io
import re
import sys
import unittest
from contextlib import redirect_stdout
from kerasRun import makeNumeric
from kerasRun import predict
from kerasRun import readData
from kerasRun import run10cv
from kerasRun import runExperiment

class myTest(unittest.TestCase):
    def testMakeNumeric(self): pass

    def testPredict(self): pass

    def testReadData(self): pass

    def testRun10cv(self): pass

    def testRunExperiment(self): pass
         
if __name__ == '__main__':
    unittest.main()
