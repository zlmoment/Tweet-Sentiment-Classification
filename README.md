## Tweet Sentiment Classification

This is the project of my course ECE/CS 7220. Many thanks to Dr. DeSouza, who is the intructor of this course. 

It is implemented using NLTK and Scikit-learn. KNN and Naive Bayes are used. ~~I will open source it when my class is finished.~~ The class is finished, so I open source my code.

If you find a bug, please let me know: Zhaoyu Li (zlmoment@gmail.com), or push a request on Github.

All the souce codes can only be used for learning and discussion.

(Dataset file is not provided)

### File explanation

- driver.py	
It is a driver that connects all the other Python files together and invoke functions in order.

- getFeatureList_DimRed.py	
It is used to generate the vocabulary from the training set, this file uses Chi-Square test to do feature selection.

- getFeatureList_DimRed2.py	
It is the same as above file but using term vector to do feature selection.

- classify.py	
It is used to preprocess the tweet, build and train the classifier, and do cross validation.

- utils.py	
It is a collection of useful functions that may be invoked multiple times in other Python files.