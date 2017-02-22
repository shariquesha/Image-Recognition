import cPickle as pickle
import os
import numpy as np
#import bob
#import bob.io.base
#import ipython
#import ipy_autoreload

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

#%matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')

#%load_ext autoreload
#%autoreload 2
#ipython = get_ipython()

#if '__IPYTHON__' in globals():
#    ipython.magic('load_ext autoreload')
 #   ipython.magic('autoreload 2')

layers0 = [
    # layer dealing with the input data
    (layers.InputLayer, {'shape': (None, 3, 150, 150)}),

    # first stage of our convolutional layers
    (layers.Conv2DLayer, {'num_filters': 96, 'filter_size': (7,7),'stride':(2,2)}),
    (layers.BatchNormLayer,{}),
#     (layers.normalization),
    (layers.MaxPool2DLayer, {'pool_size': (2,2)}),

    # second stage of our convolutional layers
    (layers.Conv2DLayer, {'num_filters': 128, 'filter_size': 5,'stride':2}),
    (layers.BatchNormLayer,{}),
#     (layers.normalization),
    (layers.MaxPool2DLayer, {'pool_size': (2,2)}),

    # third stage of our convolution layers
    (layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3}),
#     (layers.MaxPool2DLayer, {'pool_size': 2}),
    (layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3}),

    (layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3}),
    (layers.MaxPool2DLayer, {'pool_size': 2}),

#     (layers.Conv2DLayer, {'num_filters': 512, 'filter_size': 3}),
#     (layers.MaxPool2DLayer, {'pool_size': 2,'stride':2}),



    # two dense layers with dropout
    (layers.DenseLayer, {'num_units': 2048}),
    (layers.DropoutLayer, {}),
    (layers.DenseLayer, {'num_units': 1024}),

    # the output layer
    (layers.DenseLayer, {'num_units': 11, 'nonlinearity': lasagne.nonlinearities.softmax}),
]
net0 = NeuralNet(
    layers=layers0,
    update_learning_rate=0.0001,
    max_epochs=100,

    update=adam,
   

    objective_l2=0.0025,
    batch_iterator_train=BatchIterator(batch_size=60),
    batch_iterator_test=BatchIterator(batch_size=29),
    train_split=TrainSplit(eval_size=0.02),

    verbose=1,
)

