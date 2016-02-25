##Spam detection using sklearn libs
__author__ : HELAL ALI Misu 
__email__  : michou.helal@gmail.com
Descption:
	Testing multiple algo from sklearn and define the best model to use for spam detection

Usage:
-----
	Trainning:
		trainning data format : <spam/ham> \t raw message
			python detecspam.py -f <trainning_file>
	Prediction:
		python prediction.py -m <some_message>


	
Input:
-----
	Text(train),label : 
	output : model
	Text(test):

TO-DO:
----- 
	- Delete pipeline 
	- Complete DockerFile
