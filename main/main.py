from __future__ import division
# encoding: utf-8
import os
import shutil
from sklearn import svm
from picMat import LoadPic as lp
from picMat import PrePic as pp
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report

resultFolder = '../result/'
testFolder = '../test/'
trainFolder = '../train/'
p = lp.picMat(25,37,1,41,1140,trainFolder)

def createData():

    p.saveRawMatrix()    #create x_train, y_label
    p.saveTrainY()       #create y_train
    p.saveTrainData(label=True)    #create train

def preProcess(Split = True):
    # pre_processing the data , get x_train and y_train
    x_train , y_train , x_val , y_true = pp.load_TrainData(ifsplit=Split)    
    y_train = np.reshape(y_train , -1)
    y_true   = np.reshape(y_true , -1)
    
    return x_train , y_train , x_val , y_true
   
if __name__ == '__main__': 

    createData()


    x_train , y_train , x_val , y_true = preProcess()

    # standardize the data ,make mean = 0, var = 1
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_trainStd = scaler.transform(x_train)
    x_valStd   = scaler.transform(x_val)

    #SVM algorithm
    clf = svm.SVC(verbose = True)
    clf.fit(x_trainStd, y_train)

    # predict the valid data for validation
    y_pred = clf.predict(x_valStd)

    #calculate the accurate score
    print(classification_report(y_true, y_pred))

    testData = os.listdir(testFolder)

    os.chdir(testFolder)
    print 'create category folders'
    if not os.path.exists(resultFolder):
        os.mkdir(resultFolder)
    categoryList = [x[0][len(trainFolder):] for x in os.walk(trainFolder)]
    for cat in categoryList:
        if not os.path.exists(resultFolder+cat):
            os.mkdir(resultFolder + cat)

    print 'Start to categorize'

    for root in testData:

        for files in os.listdir(root+'/'):
            os.chdir(root)  
            # if os.path.isdir(ele)
            # JZ modified. because what we want is the png file, not directorys.
            #if files != 'blank' and files != 'nonBlank':
            if not os.path.isdir(files):
                x_test = p.convertMatHelper(files)/255
                
                #standardize this single x_test
                x_testStd = scaler.transform(x_test)
                y_testPred = clf.predict(x_testStd)
                
                # if y_testPred[0] == 0:
                #     shutil.move(files , 'blank')
                # else:
                #     shutil.move(files , 'nonBlank')
                #
                shutil.move(files, resultFolder + categoryList[y_testPred.index('1')])
                os.chdir(os.path.pardir)
            
            else:
                os.chdir(os.path.pardir)
                continue
    print 'Finished'