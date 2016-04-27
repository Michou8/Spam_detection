#MAINTENAIR misu.helal@gmail.com
FROM ubuntu

#File / Author Maintainer
MAINTAINER Kwyn Meagher

#Update repositor source list
RUN sudo apt-get update

################## BEGIN INSTALLATION ######################
#Install python basics
RUN apt-get -y install \
	build-essential \
	python-dev \
	python-setuptools \
	python-pip

#Install scikit-learn dependancies
RUN apt-get -y install \
	python-numpy \
	python-scipy \
	libatlas-dev \
	libatlas3-base

#Install textblob and pandas
RUN pip install textblob 
RUN pip install pandas

#Install Django
RUN pip install django

#Install scikit-learn
RUN apt-get -y install python-sklearn

# Add spam detection server
ADD . /src/

#Run spam detection server
CMD "python" "/src/spam_api/manager.py runserver"

