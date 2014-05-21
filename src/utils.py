#!/usr/bin/python
# -*- coding: utf-8 -*-

# ECE/CS 7720
# @author: Zhaoyu Li

#
# a utils collection to provide some helper function
#

import csv
import re
import nltk
import string
from nltk.corpus import stopwords

# preprocess tweet
def processTweet(tweet):
	# Convert to lower case
	tweet = tweet.lower()
	# Remove www.* or https?://*
	tweet = re.sub(r'((www\.[\s]+)|(https?://[^\s]+))','',tweet)
	# Remove @username
	tweet = re.sub(r'@[^\s]+','',tweet)
	# Remove additional white spaces
	tweet = re.sub(r'[\s]+', ' ', tweet)
	# Replace #word with word
	tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
	# Romove 2 or more repetitions of character and replace with the character itself repetitions of character and replace with the character itself
	tweet = re.sub(r"(.)\1{1,}", r"\1\1", tweet)
	# Remove words start with a digit
	tweet = re.sub(r"[0-9][a-zA-Z0-9]*",'',tweet)

	# trim
	tweet = tweet.strip('\'"')

	# Remove punctuationn
	for punct in string.punctuation:
		tweet = tweet.replace(punct,'')

	# Remove stopwords & length <= 2
	tweet = [w.lower() for w in nltk.word_tokenize(tweet) if not w in stopwords.words('english') and len(w) > 2]

	return tweet

# get the total number of tweet in a file
def getAmount(fileaddress):
	numPos = 0
	numNeg = 0

	with open(fileaddress,'rb') as csvfile:
		filereader = csv.DictReader(csvfile, delimiter=',')
		for row in filereader:
			if row['class'] == '0':
				numPos += 1
			else:
				numNeg += 1

	return numPos+numNeg

# print info on scree
def log(str):
	print "[INFO] " + str
	#pass

# print important info on screen
def implog(str):
	print "[>>>>] " + str + " [<<<<]"