import cPickle as pickle
import os
import numpy as np
import bob
import bob.io.base

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import lasagne
from lasagne import layers
from lasagne.updates import adam
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

from nolearn.lasagne import BatchIterator
# from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

%matplotlib inline
%load_ext autoreload
%autoreload 2
