#!/usr/bin/python
# -*- coding: utf-8 -*-

# ECE/CS 7720
# @author: Zhaoyu Li
# @mail: zlht3@mail.missouri.edu


import csv
import sys
import re
import itertools
import nltk
import string
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import stopwords

from utils import *

import operator

# the main class
class getFeatureList_TF():
	"""
	get the feature list using term frequency
	the only differnce between this file and getFeatureList_DimRed.py
	is this file uses term frequency
	"""
	trainingDataFile = ""
	feature_list = []
	num = 0

	def __init__(self, trainingDataFile, num):
		self.trainingDataFile = trainingDataFile
		self.num = num

	# get all the words collection
	def getAllWords(self):
		csvfile = open(self.trainingDataFile,'rb')
		filereader = csv.DictReader(csvfile, delimiter=',')
		posWords = []
		negWords = []
		for row in filereader:
			if row['class'] == '4':
				posWords.append(processTweet(row['text']))
			else:
				negWords.append(processTweet(row['text']))
		return [posWords, negWords]

	# based on term frequency, compute the score for each word
	def create_word_scores(self):
		[posWords, negWords] = self.getAllWords()
		
		posWords = list(itertools.chain(*posWords))
		negWords = list(itertools.chain(*negWords))

		words = []
		words.extend(posWords)
		words.extend(negWords)

		freq = {}

		for word in words:
			if word in freq.keys():
				freq[word] = freq[word] + 1
			else:
				freq[word] = 1
		return freq

	# find the best k words
	def find_best_words(self, word_scores):
		sorted_word_scores = sorted(word_scores.iteritems(), key=operator.itemgetter(1))
		sorted_word_scores.reverse()
		ret = []
		i = 0
		for word, v in sorted_word_scores:
			if i < int(self.num):
				ret.append(word)
				i += 1
			else:
				break
		return ret

	# write feature list to a file for future uses
	def writeOutput(self, filename, writeOption='w'):
		fp = open(filename, writeOption)
		for w in self.feature_list:
			writeStr = w + ","
			fp.write(writeStr)
		fp.write('\n')
		fp.close()

	# the main loop
	def generateFeatureList(self):
		log("Generating word scores...")
		word_scores = self.create_word_scores()
		log("Finding best words_list...")
		self.feature_list = self.find_best_words(word_scores)

		log("Writing to output file...")
		self.writeOutput("feature_list.data")

		log("Done.")

# the main entrance
if __name__ == '__main__': 
	if len(sys.argv) != 3:
		log("Please add training data file, and words number")
		exit()
	# get the address of training data
	trainingDataFile = sys.argv[1]
	# get the number of samples
	num = sys.argv[2]
	# create class and start to generate feature list
	cl = getFeatureList_TF(trainingDataFile, num)
	cl.generateFeatureList()