import cPickle as pickle
import cPickle as pickle
import os
import numpy as np


# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import lasagne
from lasagne import layers
from lasagne.updates import adam
from lasagne.updates import nesterov_momentum

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit

from nolearn.lasagne import BatchIterator
from nolearn.lasagne import visualize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


f=open("imageclassification.pkl","rb")
data=pickle.load(f);

X_train=[] #training image names  in X_train
Y_train=[] #trainig image labels in Y_train

index =0
str1 = "No Indexing"

for value in data['img_detail']:
	if value :
		if str1.find(value['major'][0]) :
			X_train.append(data['img_name'][index])
			Y_train.append(value['major'][0])
	index+=1

'''count =1 
for i in X_train:
	print count,i + "train"
	count+=1'''
f.close()
Y_train=np.asarray(Y_train)
X_train=np.asarray(X_train)

########################## Network setup ##############################

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
    batch_iterator_train=BatchIterator(path='/home/sharique/Documents/minor/dataset/',batch_size=60),
    batch_iterator_test=BatchIterator(path='/home/sharique/Documents/minor/dataset/',batch_size=29),
    train_split=TrainSplit(eval_size=0.02),

    verbose=1,
)

#################################################################
# bi = BatchIterator(batch_size=29)
#################################################################

# from nolearn.lasagne import PrintLayerInfo

# layer_info=PrintLayerInfo()
#################################################################
# type(Y_train)
print max(Y_train)
# Y_train=np.asarray(Y_train)
# X_train=np.asarray(X_train)
#Y_train=Y_train.astype(np.int32)
###################################################################


print X_train.shape
print Y_train.shape

###################################################################
net0.fit(X_train,Y_train)
###################################################################



import pydot

visualize.draw_to_notebook(net)

###################################################################
import cv2
for Xb,yb in bi(X_train,y_train):
    a=1
#     print type(Xb)
# all the input array dimensions except for the concatenation axis must match exactly

print Xb


