#!/usr/bin/python
# -*- coding: utf-8 -*-

# ECE/CS 7720
# @author: Zhaoyu Li
# @mail: zlht3@mail.missouri.edu

import os
import sys
from classify import *
from getFeatureList_ChiSquare import *

# get input arguments
s = int(sys.argv[1])
d = int(sys.argv[2])

print "Sampels:%d *2  Dim:%d" % (s,d)

# get samples from the whole dataset
os.system('head -n %d training.csv > training.test.csv;' % int(s+1))
os.system('tail -n %d training.csv >> training.test.csv;' % int(s))

# get feature list (vocabulary)
trainingDataFile = "training.test.csv"

# if you are going to use term frequency, you need to use next line

# if you are going to use term frequency, you need to use next line
cl = getFeatureList_ChiSquare(trainingDataFile, d)
cl.generateFeatureList()

# trainign and classify
cl = Classify(trainingDataFile)
cl.classify()