from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from django.http import JsonResponse
import urllib
import json
import os
import csv
from textblob import TextBlob
import cPickle
import numpy as np
# TO-DO : log event file
# import the logging library
#import logging
# Get an instance of a logger
#logger = logging.getLogger(__name__)

# Get the right path to the spam detection model
SPAM_DETECTION_MODEL_PATH = "{base_path}/spam_model/sms_spam_detector.pkl".format(base_path=os.path.abspath(os.path.dirname(__file__)))
model = open(SPAM_DETECTION_MODEL_PATH)
#Load the model in memory
svm_detector_reloaded = cPickle.load(model)
# This fix the insecure error
model.close()

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
def CounterWords(messages):
        c = 0
	data = []
        for message in messages:
                tmp = {}
		message = split_into_lemmas(message)
                for word in message:
                        if word not in tmp:
                                tmp[word] = 1
                        else:
                                tmp[word] += 1

		data.append(tmp)
	return data

FILE_VECTOR = 'cvector.json'
with open(FILE_VECTOR,'rb') as f:
	model = json.load(f)

FILE_VECTOR_MODEL = "vector.pkl"
data = model['_Xdata']
dt = DictVectorizer()
message_bow = dt.fit_transform(data)
tfidf_transformer = TfidfTransformer().fit(message_bow)
messages_tfidf = tfidf_transformer.fit_transform(message_bow)
tf_idf = Pipeline([('dict', dt),('tfidf', tfidf_transformer)])


 
@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
 
	# check to see if this is a post request
	if request.method == "POST":
		# Read the content into the request
		message = request.read()
		data = tf_idf.transform(CounterWords([message.lower()]))
		message = data
		classes = svm_detector_reloaded.classes_
		prediction_label = ''
		proba = svm_detector_reloaded.predict_proba(message)
		proba = proba[0]
		p = 0.0
		p_d= {}
		for i in xrange(len(classes)):
			p_d[classes[i]] = proba[i]
		
		if proba[0] > proba[1]:
			prediction_label = classes[0]
		elif proba[1] > proba[0] :
			prediction_label = classes[1]
		else:
			prediction_label = 'spam'
			
		# Predict the category of this message
		data = {"prediction":prediction_label,'proba_spam':p_d['spam']}
		print data
	# return a JSON response
	return JsonResponse(data)
