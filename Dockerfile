From ubuntu:14.04
#MAINTENAIR misu.helal@gmail.com
RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update
RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy upgrade
RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy dist-upgrade

RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy install build-essential gfortran graphviz-dev imagemagick libatlas-dev libatlas3-base libfreetype6-dev liblapack-dev libpng-dev libxml2-dev libxslt1-dev libyaml-dev pandoc pkg-config python-dev python-pip supervisor zlib1g-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy autoremove
RUN DEBIAN_FRONTEND=noninteractive apt-get -qqy autoclean
RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install bottleneck cython numexpr nose patsy pyenchant pygments pygraphviz pytz pyyaml
RUN pip install configobj lxml python-dateutil networkx textblob
RUN pip install beautifulsoup4 gensim ipython[notebook] mpltools nltk pandas pattern scikit-learn simpy ujson
RUN pip install statsmodels
RUN pip install seaborn
RUN pip install textblob
RUN pip install cPickle
