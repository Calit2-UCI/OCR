from __future__ import division
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
theano.config.floatX = 'float32' #use this config and np.array(dtype='float32') to enable gpu
#import lasagne
from lasagne import layers, nonlinearities
from lasagne import objectives
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
#from nolearn.dbn import DBN
#from boto.cloudformation.stack import Output
from sklearn.metrics import classification_report, regression
from lasagne.nonlinearities import softmax

#######add by jz
import shutil
import processpic as pp
import platform
import neuralNet as nn

def trainningProcess(model='MLP'):
    print "Start training:"
    # ###########training process #####################
    picmat = pp.picMat()
    picmat.saveRawMatrix()  # create x_train, y_label
    picmat.saveTrainY()  # create y_train
    picmat.saveTrainData()  # create train

    if (model == 'MLP'):
        x_train, y_train = nn.load_TrainData()
        net = nn.build_mlp()
        net.fit(x_train, y_train)
        net.save_params_to('MLP_weights_file')

    elif (model == 'CNN'):
        x_train, y_train = nn.load_TrainData2D()
        ###########training process #####################
        net = nn.buildCNN()
        net.fit(x_train, y_train)
        plot_loss(net)
        plot_conv_weights(net.layers_[1], figsize=(4, 4))
        plt.show()
        net.save_params_to('CNN_weights_file')

    else:
        print 'ERROR: Please select a model #MLP or #CNN !'

    ############ end of training process #################

    return net, picmat


######JZ added functions below for categorizing images into different folders in order to get training data easier

resultFolder = 'result/'
testFolder = 'test/'
trainFolder = 'train/'

def createCategorizedFolders():
    print 'create category folders'
    if os.path.exists(resultFolder):
        shutil.rmtree(resultFolder)
    os.mkdir(resultFolder)
    categoryList = [x[0][len(trainFolder):] for x in sorted(os.walk(trainFolder))][1:]
    for cat in categoryList:
        if not os.path.exists(resultFolder + cat):
            os.mkdir(resultFolder + cat)
    print 'creating folders succeeded'
    return categoryList

def categorize(net, picmat, categoryList, model='CNN'):
    print 'start categorizing images'
    testData = os.listdir(testFolder)
    testData.sort()
    for root in testData:
        curPath = testFolder + root + '/'
        print 'curent path: ', curPath
        picmat.saveRawTestData(curPath)
        if model == 'MLP':
            x_test, y_label = nn.load_TestData()
        elif model == "CNN":
            x_test, y_label = nn.load_TestData2D()
        else:
            print 'ERROR: Please select a model #MLP or #CNN !'
            return
        y_pred = net.predict(x_test)
        dataList = sorted(os.listdir(curPath))

        for i in range(y_pred.shape[0]):
             shutil.copy(curPath+dataList[i], resultFolder + categoryList[y_pred[i].tolist().index(max(y_pred[i]))])
    print 'categorizing succeeded'


def main(model='CNN'):
    # note:
    # 1. still finding a way to get the test data directory instand of putting
    # them in a csv file and read again.
    print "using model: ", model
    net, picmat = trainningProcess(model)
    categoryList= createCategorizedFolders()
    categorize(net, picmat, categoryList, model = model)

    # print 'delete original directory'
    # shutil.rmtree(testFolder)
if __name__ == '__main__':
    # main(model="MLP")
    # print(platform.system())
    if(platform.system()== "Darwin"):
        os.system("find . -name '.DS_Store' -type f -delete")
    p = pp.picMat()

    # p.saveRawMatrix()  # create x_train, y_label
    # p.saveTrainY()  # create y_train
    # p.saveTrainData()  # create train
    # p.saveRawTestData("test/")  # create x_test, y_testLabel
    main(model='CNN')



# thresholding: seems to be similar. see figure
#dropout: seem to get a better result, see figures
# add training data: get a better result. see figures
# increase epochs: trn/val become worse after 165. it will drop to 0.2 at 300. However, after adding training data,
#                   I get a much better rate.


# for mac users before running this program:
# in terminal cd to the OCR directory and do:
# find . -name '.DS_Store' -type f -delete