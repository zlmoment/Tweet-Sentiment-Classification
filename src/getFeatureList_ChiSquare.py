#!/usr/bin/python
# -*- coding: utf-8 -*-

# ECE/CS 7720
# @author: Zhaoyu Li
# @mail: zlht3@mail.missouri.edu

#
# get the feature list using Chi-Square test
#

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
class getFeatureList_ChiSquare():

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

	# based on Chi-Square, compute the score for each word
	def create_word_scores(self):
		[posWords, negWords] = self.getAllWords()
		
		posWords = list(itertools.chain(*posWords))
		negWords = list(itertools.chain(*negWords))

		word_fd = FreqDist()
		cond_word_fd = ConditionalFreqDist()
		for word in posWords:
			word_fd.inc(word)
			cond_word_fd['pos'].inc(word)
		for word in negWords:
			word_fd.inc(word)
			cond_word_fd['neg'].inc(word)

		pos_word_count = cond_word_fd['pos'].N()
		neg_word_count = cond_word_fd['neg'].N()
		total_word_count = pos_word_count + neg_word_count

		log("Total number of words: %d" % total_word_count)

		word_scores = {}
		for word, freq in word_fd.iteritems():
			pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
			neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
			word_scores[word] = pos_score + neg_score

		return word_scores

	# find the best k words
	def find_best_words(self, word_scores):
		number = int(self.num)
		best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
		best_words_list = set([w for w, s in best_vals])
		log("Found %d" % number)
		return best_words_list

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
	cl = getFeatureList_ChiSquare(trainingDataFile, num)
	cl.generateFeatureList()