#import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
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
########### Funciton #################
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

(options, args) = parser.parse_args()

FILE_TRAIN = options.filename
################
# You can connect to sql database
"""df = psql.read_sql(('select "Timestamp","Value" from "MyTable" '
                     'where "Timestamp" BETWEEN %(dstart)s AND %(dfinish)s'),
                   db,params={"dstart":datetime(2014,6,24,16,0),"dfinish":datetime(2014,6,24,17,0)},
                   index_col=['Timestamp'])
"""
messages = pandas.read_csv(FILE_TRAIN, sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])


bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(messages['message'])
messages_bow = bow_transformer.transform(messages['message'])
tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)

classe = ['ham', 'spam']

if 'nb_model.pkl' not in os.listdir("./"):
	print 'First trainning'
	nb = MultinomialNB()
	classe = ['ham', 'spam']
	nb.partial_fit(messages_tfidf, messages['label'],classes=classe)
	all_predictions = nb.predict(messages_tfidf)
	msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.3)
	print classification_report(messages['label'], all_predictions)
	# store the spam detector to disk after training
	with open('nb_model.pkl', 'wb') as fout:
	    cPickle.dump(nb, fout)
else:
	print "Training with partial_fit"
	with open('nb_model.pkl','rb') as f:
		nb = cPickle.load(f)
	nb.partial_fit(messages_tfidf, messages['label'],classes=classe)
        all_predictions = nb.predict(messages_tfidf)
        print classification_report(messages['label'], all_predictions)
        # store the spam detector to disk after training
        with open('nb_model.pkl', 'wb') as fout:
            cPickle.dump(nb, fout)

