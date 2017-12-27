import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split

def processData():
	card_data.set_index('Time', inplace = True)
	card_data.drop('Amount', inplace = True)

def predictSVM():
	X = np.array(card_data.drop(['Class'],1))
	X = preprocessing.scale(X)
	y = np.array(card_data['Class'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	clf = svm.SVC(kernel = 'linear')
	clf.fit(X_train, y_train)

	return clf.score(X_test, y_test)

#Data is in pickled format
card_data = pd.read_csv('creditcard.csv')
card_data.to_pickle('card.pickle')
card_data = pd.read_pickle('card.pickle')

processData()

score = predictSVM() # predict the score using SVM classifier

print('The accuracy of detecting fradulent card using SVM is', score)


