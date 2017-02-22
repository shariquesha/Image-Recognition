layers0 = [
    # layer dealing with the input data
    (layers.InputLayer, {'shape': (None, 3, 200, 200)}),

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
