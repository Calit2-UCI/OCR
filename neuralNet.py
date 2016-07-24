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

# import theano
# theano.config.device = 'cpu'
# theano.config.floatX = 'float32' #use this config and np.array(dtype='float32') to enable gpu
#import theano.tensor as T 
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

resultFolder = 'result/'
testFolder = 'testBlank/'
trainFolder = 'train/'
########end of add by jz

#### load traindata
#    split it into train data and validation data
#    return x_train, y_train, x_val, y_val

def load_TrainData(filename='train.csv',classNumber=52,ifsplit=False,valPrcnt=0.2):
    train = pd.read_csv(filename)
   # print train.shape
    train = train.iloc[:,1:]
    #print train.iloc[:,1:]
    x_train = train.iloc[:,classNumber:]
    y_train = train.iloc[:,:classNumber]
    #print y_train.shape
    if (ifsplit):
        valNum = int(x_train.shape[0]*valPrcnt)
    
        x_train, x_val = x_train.iloc[:-valNum,:], x_train.iloc[-valNum:,:]
        y_train, y_val = y_train.iloc[:-valNum,:], y_train.iloc[-valNum:,:]
        
        return np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val)
    
    else:
        return np.array(x_train),np.array(y_train)

#### a three-dimensional matrix with shape (c, 0, 1),
#    where c is the number of channels (colors),
#    and 0 and 1 correspond to the x and y dimensions 
#    of the input image. In our case, the concrete shape will be (1, 96, 96),
#    because we're dealing with a single (gray) color channel only.

def load_TrainData2D(filename='train.csv',classNumber=52,width=25, height=37,ifsplit=False,valPrcnt=0.2):
    train = pd.read_csv(filename)
    train = train.iloc[:,1:]

    x_train = np.array(train.iloc[:,classNumber:])
    y_train = np.array(train.iloc[:,:classNumber])
    
    if (ifsplit):
        valNum = int(x_train.shape[0]*valPrcnt)
    
        x_train, x_val = x_train.iloc[:-valNum,:], x_train.iloc[-valNum:,:]
        y_train, y_val = y_train.iloc[:-valNum,:], y_train.iloc[-valNum:,:]
        
        x_train = x_train.reshape(-1,1,height,width)
        x_val = x_val.reshape(-1,1,height,width)
        
        return x_train, y_train, x_val, y_val
    else:
    #reshape x into (1,height,width)  # '1' represents 1 channel
        x_train = x_train.reshape(-1,1,height,width)
    #y_train = y_train.astype(np.int32).reshape(320,)-1
    
        return x_train,y_train

def load_TestData(filename_X='x_test.csv',filename_Y='y_testLabel.csv',width=25, height=37):
    x_test = np.array(pd.read_csv(filename_X).iloc[:,1:]/255)
    y_test = np.array(pd.read_csv(filename_Y).iloc[:,1:])
    return x_test,y_test





def load_TestData2D(filename_X='x_test.csv',filename_Y='y_testLabel.csv',width=25, height=37):
    x_test = np.array(pd.read_csv(filename_X).iloc[:,1:]/255)
    y_test = np.array(pd.read_csv(filename_Y).iloc[:,1:])

    x_test = x_test.reshape(-1,1,height,width)

    return x_test,y_test


# def load_TestData2D_X(filename_X='x_test.csv', width=25, height=37):
#     x_test = np.array(pd.read_csv(filename_X).iloc[:, 1:] / 255)
#     x_test = x_test.reshape(-1, 1, height, width)
#
#     return x_test


#### build custom MLP neural network

def build_mlp():
    network = NeuralNet(
        layers=[#threelayers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output',layers.DenseLayer),     
            ],
        #layer parameters            
        input_shape = (None,25*37),  #25*37 input pixels per batch
        hidden_num_units = 800,      #number of units in hidden layer
        output_nonlinearity=softmax,  #output use identity function             
        # output_num_units=41,          #8 classes(target)
        output_num_units=52,
        #optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,
        
        regression=True,
        max_epochs=500, #we want to train this 400 times
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
        dropout4_p=0.5,
    hidden5_num_units=500,
    output_num_units=52,
    # output_num_units=41,
    output_nonlinearity=softmax,

    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,



    objective_loss_function=objectives.categorical_crossentropy,
    ##########
    #   objectives.binary_accuracy is for threshold, not working. dont know how to implement
    #   http://lasagne.readthedocs.io/en/latest/modules/objectives.html
    #   lasagne.objectives.binary_accuracy(predictions, targets, threshold=0.5)
    #
    # objective_evaluation_function = objectives.binary_accuracy,
    ##########
    regression=True,
    max_epochs=300,
    verbose=1,
    )
    
    return net2

def singleFolderCategorize(classNumber=41):
    pass

def trainningProcess(model='MLP'):
    print "Start training:"
    # ###########training process #####################
    picmat = pp.picMat()
    picmat.saveRawMatrix()  # create x_train, y_label
    picmat.saveTrainY()  # create y_train
    picmat.saveTrainData()  # create train

    if (model == 'MLP'):
        x_train, y_train = load_TrainData()
        net = build_mlp()
        net.fit(x_train, y_train)
    elif (model == 'CNN'):
        x_train, y_train = load_TrainData2D()
        net = buildCNN()
        net.fit(x_train, y_train)
        plot_loss(net)
        plot_conv_weights(net.layers_[1], figsize=(4, 4))
        plt.show()
    else:
        print 'ERROR: Please select a model #MLP or #CNN !'

    ############ end of training process #################
    return net, picmat



def createCategorizedFolders(resultFolder ='result/' ):
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

def categorize(net, picmat, categoryList, testFolder = 'testBlank/',model='MLP'):
    print 'start categorizing images'
    testData = os.listdir(testFolder)
    testData.sort()
    for root in testData:
        curPath = testFolder + root +'/nonBlank'+ '/'
        print 'curent path: ', curPath
        picmat.saveRawTestData(curPath)
        if model == 'MLP':
            x_test, y_label = load_TestData()
        elif model == "CNN":
            x_test, y_label = load_TestData2D()
        else:
            print 'ERROR: Please select a model #MLP or #CNN !'
            return
        y_pred = net.predict(x_test)
        dataList = sorted(os.listdir(curPath))

        for i in range(y_pred.shape[0]):
             shutil.copy(curPath+dataList[i], resultFolder + categoryList[y_pred[i].tolist().index(max(y_pred[i]))])
    print 'categorizing succeeded'


    pd.DataFrame(y_pred).to_csv('y_pred_{}.csv'.format(model))
    y = []

    for i in range(y_pred.shape[0]):
        y.append(y_pred[i].tolist().index(max(y_pred[i])) + 1)

    pd.DataFrame(y).to_csv('y_predlabel_{}.csv'.format(model))
    print classification_report(y_label, y)



def main(model='MLP'):
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
    main(model='CNN')
# thresholding and dropout
#dropout: seem to get a better result, see figures
# add training data
# fix MLP:  it will be pending after training because no figures are available for this one.
# increase epochs: trn/val become worse after 165. it will drop to 0.2 at 300
    