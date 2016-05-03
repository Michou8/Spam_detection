from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import urllib
import json
import os
import csv
from textblob import TextBlob
import cPickle

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
 
@csrf_exempt
def detect(request):
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
 
	# check to see if this is a post request
	if request.method == "POST":
		# Read the content into the request
		message = request.read()
		# Predict the category of this message
		data = {"prediction":svm_detector_reloaded.predict([message])[0],'proba':str(svm_detector_reloaded.predict_proba([message]))}
	# return a JSON response
	return JsonResponse(data)
