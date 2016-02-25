import csv
from textblob import TextBlob
import cPickle

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
parser.add_option("-m", "--msg", dest="message",help="Message to identify if ham or spam",default='SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info')
(options, args) = parser.parse_args()

MESSAGE_ = options.message
model = open('sms_spam_detector.pkl')
#svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))
svm_detector_reloaded = cPickle.load(model)
model.close()
message = MESSAGE_ 
print 'Prediction:\t', svm_detector_reloaded.predict([message])[0]
