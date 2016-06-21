import logging
 
from logging.handlers import RotatingFileHandler
 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
 
formatter = logging.Formatter('{"date": %(asctime)s, "type": %(levelname)s,%(message)s}')
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
 
steam_handler = logging.StreamHandler()
steam_handler.setLevel(logging.DEBUG)
logger.addHandler(steam_handler)

def log():
	return logger 
