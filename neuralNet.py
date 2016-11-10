import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

import os
import os.path
import glob
import time

from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion

import theano

# theano.config.device = 'gpu'
theano.config.floatX = 'float32'  # use this config and np.array(dtype='float32') to enable gpu
# import theano.tensor as T
# import lasagne
from lasagne import layers, nonlinearities
from lasagne import objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
# from nolearn.dbn import DBN
# from boto.cloudformation.stack import Output
from sklearn.metrics import classification_report, regression
from lasagne.nonlinearities import softmax
from processpic import picMat

#### load traindata
#    split it into train data and validation data
#    return x_train, y_train, x_val, y_val

def load_TrainData(filename='train.csv', classNumber=52, ifsplit=False, valPrcnt=0.2):
    train = pd.read_csv(filename)
    # print train.shape
    train = train.iloc[:, 1:]
    # print train.iloc[:,1:]
    x_train = train.iloc[:, classNumber:]
    y_train = train.iloc[:, :classNumber]
    # print y_train.shape
    if (ifsplit):
        valNum = int(x_train.shape[0] * valPrcnt)

        x_train, x_val = x_train.iloc[:-valNum, :], x_train.iloc[-valNum:, :]
        y_train, y_val = y_train.iloc[:-valNum, :], y_train.iloc[-valNum:, :]

        return np.array(x_train, dtype='float32'), np.array(y_train, dtype='float32'), np.array(x_val, dtype='float32'), np.array(
            y_val, dtype='float32')

    else:
        return np.array(x_train, dtype='float32'), np.array(y_train, dtype='float32')


#### a three-dimensional matrix with shape (c, 0, 1),
#    where c is the number of channels (colors),
#    and 0 and 1 correspond to the x and y dimensions 
#    of the input image. In our case, the concrete shape will be (1, 96, 96),
#    because we're dealing with a single (gray) color channel only.

def load_TrainData2D(filename='train.csv', classNumber=52, width=25, height=37, ifsplit=False, valPrcnt=0.2):
    train = pd.read_csv(filename)
    train = train.iloc[:, 1:]

    x_train = np.array(train.iloc[:, classNumber:] , dtype='float32')
    y_train = np.array(train.iloc[:, :classNumber], dtype='float32')

    if (ifsplit):
        valNum = int(x_train.shape[0] * valPrcnt)

        x_train, x_val = x_train.iloc[:-valNum, :], x_train.iloc[-valNum:, :]
        y_train, y_val = y_train.iloc[:-valNum, :], y_train.iloc[-valNum:, :]

        x_train = x_train.reshape(-1, 1, height, width)
        x_val = x_val.reshape(-1, 1, height, width)

        return x_train, y_train, x_val, y_val
    else:
        # reshape x into (1,height,width)  # '1' represents 1 channel
        x_train = x_train.reshape(-1, 1, height, width)
        # y_train = y_train.astype(np.int32).reshape(320,)-1

        return x_train, y_train


def load_TestData2D(filename_X='x_test.csv', filename_Y='y_testLabel.csv', width=25, height=37):
    x_test = np.array(pd.read_csv(filename_X).iloc[:, 1:] / 255, dtype='float32')
    # y_test = np.array(pd.read_csv(filename_Y).iloc[:, 1:])
    y_label = np.array(pd.read_csv(filename_Y).iloc[:, 1:], dtype='float32')
    x_test = x_test.reshape(-1, 1, height, width)

    return x_test, y_label#, y_test


#### build custom MLP neural network

def build_mlp():
    network = NeuralNet(
        layers=[  # threelayers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer parameters
        input_shape=(None, 25 * 37),  # 25*37 input pixels per batch
        hidden_num_units=800,  # number of units in hidden layer
        output_nonlinearity=softmax,  # output use identity function
        output_num_units=52,  # 8 classes(target)

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        max_epochs=500,  # we want to train this 400 times
        verbose=1,
    )

    return network


def buildCNN():
    net2 = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        input_shape=(None, 1, 37, 25),
        conv1_num_filters=96, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=256, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=500,
        dropout4_p=0.6,
        hidden5_num_units=500,
        output_num_units=52,
        output_nonlinearity=softmax,

        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        objective_loss_function=objectives.categorical_crossentropy,

        regression=True,
        max_epochs=230,
        verbose=1,
    )

    return net2


def main(model='MLP'):
    if (model == 'MLP'):

        x_train, y_train = load_TrainData()

        ###########training process #####################
        net = build_mlp()
        print "Start training:"
        net.fit(x_train, y_train)

        ################## calculate the precision###################

        print "======================== accuracy=========================="
        x_test = np.array(pd.read_csv('x_test.csv').iloc[:, 1:] / 255, dtype='float32')
        # y_test = np.array(pd.read_csv('y_test.csv').iloc[:,1:])
        y_label = np.array(pd.read_csv('y_testLabel.csv').iloc[:, 1:], dtype='float32')
        # print x_test
        y_pred = net.predict(x_test)
        pd.DataFrame(y_pred).to_csv('y_pred_MLP.csv')
        y = []

        for i in range(y_pred.shape[0]):
            y.append(y_pred[i].tolist().index(max(y_pred[i])) + 1)

        pd.DataFrame(y).to_csv('y_predlabel_MLP.csv')
        print classification_report(y_label, y)

        ##############save the MLP weights to file##########################
        net.save_params_to('MLP_weights_file')

    elif (model == 'CNN'):
        x_train, y_train = load_TrainData2D()
        # print x_train.shape
        # print y_train
        ###########training process #####################
        net = buildCNN()
        print "Start training:"
        net.fit(x_train, y_train)

        plot_loss(net)
        plot_conv_weights(net.layers_[1], figsize=(4, 4))
        plt.show()
        ################## calculate the precision###################

        print "======================== accuracy=========================="
        x_test, y_label = load_TestData2D()
        # x_test, y_test = load_TestData2D();
        #y_label = np.array(pd.read_csv('y_testLabel.csv').iloc[:, 1:], dtype='float32')

        y_pred = net.predict(x_test)
        pd.DataFrame(y_pred).to_csv('y_pred_CNN.csv')
        y = []

        for i in range(y_pred.shape[0]):
            y.append(y_pred[i].tolist().index(max(y_pred[i])) + 1)

        pd.DataFrame(y).to_csv('y_predlabel_CNN.csv')
        print classification_report(y_label, y)

        ##############save the CNN weights to file##########################
        net.save_params_to('CNN_weights_file')

    else:
        print 'ERROR: Please select a model #MLP or #CNN !'


if __name__ == '__main__':
    p = picMat()

    p.saveRawMatrix()  # create x_train, y_label
    p.saveTrainY()  # create y_train
    p.saveTrainData()  # create train
    p.saveRawTestData("test/")  # create x_test, y_testLabel
    main(model='CNN')



