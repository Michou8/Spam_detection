Spam detection:
==============
Testing multiple algo from sklearn and define the best model to use for spam detection

Requirement:
-----------
	- pip install -r requirement.txt
	- python -m textblob.download_corpora (It will be optional)

Usage:
-----
	Non api mode:
		- Trainning:
			- trainning data format : <spam/ham> \t raw message
				python detecspam.py -f <trainning_file>
		-Prediction:
			python prediction.py -m '<some_message>'
	API mode:
		- Start django server
			python manage.py runserver
		- prediction:
			curl -X POST 'http://localhost:8000/spam_detection/detect/' -d 'YOUR_MESSAGE'			
			Example:
				curl -X POST 'http://localhost:8000/spam_detection/detect/' -d "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"

	
Input:
-----
	Text(train),label : 
	output : model
	Text(test):

TO-DO:
----- 
	- Delete pipeline 
	- Complete DockerFile
__author__ : HELAL ALI Misu
__email__  : michou.helal@gmail.com

