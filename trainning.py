#import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
import os
import testLog
import sqlite3
import datetime
########### Function #################
def split_into_tokens(message):

    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]
######################################
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-f", "--file", dest="filename",help="Train data file  [label \t message]",default='./data/SMSSpamCollection')
#parser.add_option("-q", "--quiet",action="store_false", dest="verbose", default=True,help="don't print status messages to stdout")
parser.add_option("-t", "--train", dest="trainning",help="Train file path",default='spam_api/spam_detector/spam_model/sms_spam_detector.pkl')
(options, args) = parser.parse_args()

FILE_TRAIN = options.filename
FILE_TRAIN_LOCATION = options.trainning
FILE_VECTOR_MODEL = 'spam_api/vector.pkl'
FILE_VECTOR = 'spam_api/cvector.json'
FILE_DATABASE = 'spam_api/db.sqlite3'

conn = sqlite3.connect(FILE_DATABASE)
c_sql = conn.cursor()
# Create table
# It's not necessary the better solution try catch
try:
	c_sql.execute('''CREATE TABLE T_SPAM_LEARNING
	             (date text, content text,proba_spam real,supervised_utils real,processed_model real)''')
except:
	print 'T_SPAM_LEARNING exist'

messages = pandas.read_csv(FILE_TRAIN, sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])
model = {}
def cCounterWords(messages,label,model={}):
	c = 0
	SIZE_NEW_DATA = 0

	if '_Xlabel' in model:
		SIZE_NEW_DATA =  len(model['_Xlabel'])
	testLog.log().info('"size_data" :'+str(SIZE_NEW_DATA))
	print SIZE_NEW_DATA
	for message in messages:
		# Do this instead
		#t = ('RHAT',)
		#c.execute('SELECT * FROM stocks WHERE symbol=?', t)
		if label[c] == 'spam':
			p = 1
		else:
			p = 0
		message = split_into_lemmas(message)
		try:
                        t = (str(datetime.datetime.now()),str(message),p,1,1,)
                        c_sql.execute("INSERT INTO T_SPAM_LEARNING VALUES (?,?,?,?,?)",t)
                except:
                        print 'unicode present'
		
		tmp = {}
		for word in message:
			if word not in tmp:
				tmp[word] = 1
			else:
				tmp[word] += 1
		if '_Xdata' not in model:
			model['_Xdata'] = []
			model['_Xlabel'] = []

		# Except double message
		if tmp not in model['_Xdata']:
			model['_Xdata'].append(tmp)
			model['_Xlabel'].append(label[c])
		c += 1
	# Dump message to futur trainning
	diff = len(model['_Xlabel']) - SIZE_NEW_DATA
	testLog.log().info('"size_data" :'+str(len(model['_Xlabel'])))
	SIZE_NEW_DATA = len(model['_Xlabel']) - diff
	
	with open(FILE_VECTOR,'wb') as f:
		json.dump(model,f)
	return model,SIZE_NEW_DATA


if FILE_VECTOR in  os.listdir('./'):
	print FILE_VECTOR
	with open(FILE_VECTOR,'rb') as f:
		model = json.load(f)



	
model,SIZE_NEW_DATA = cCounterWords(messages['message'],messages['label'],model=model)
print SIZE_NEW_DATA

from sklearn.feature_extraction import DictVectorizer
def vectorization(data):
	dt = DictVectorizer()
	message_bow = dt.fit_transform(data)
	testLog.log().info('"size_vocab" :'+str(message_bow.shape[1]))
	tfidf_transformer = TfidfTransformer().fit(message_bow)
	messages_tfidf = tfidf_transformer.fit_transform(message_bow)
	tf_idf = Pipeline([('dict', dt),('tfidf', tfidf_transformer)])
	# Dump the vectorization method for testing new massage
	with open(FILE_VECTOR_MODEL,'wb') as f:
		cPickle.dump(tf_idf, f)
	return messages_tfidf
data = model['_Xdata']

# Put all classes their
classe = ['ham', 'spam']

if FILE_TRAIN_LOCATION not in os.listdir("./"):
	print 'First trainning'
	nb = MultinomialNB()
	messages_tfidf = vectorization(data)
	nb.partial_fit(messages_tfidf, model['_Xlabel'],classes=classe)
	all_predictions = nb.predict(messages_tfidf)
	#msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
	print classification_report(model['_Xlabel'], all_predictions)

	# store the spam detector to disk after training
	with open(FILE_TRAIN_LOCATION, 'wb') as fout:
	    cPickle.dump(nb, fout)
else:
	print "Training with partial_fit"
	# With sckit learn new feature not be allowed to appear

	#with open(FILE_TRAIN_LOCATION,'rb') as f:
	#	nb = cPickle.load(f)
	nb = MultinomialNB()
	print SIZE_NEW_DATA
	if SIZE_NEW_DATA != len(data):
		messages_tfidf = vectorization(data)
		messages = messages_tfidf
		label = model['_Xlabel']
		nb.partial_fit(messages, label,classes=classe)
	        all_predictions = nb.predict(messages_tfidf)
		
	        print classification_report(model['_Xlabel'], all_predictions)
		# store the spam detector to disk after training
		with open(FILE_TRAIN_LOCATION, 'wb') as fout:
			cPickle.dump(nb, fout)

