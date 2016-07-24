from __future__ import division
import os
import os.path
import shutil
import numpy as np
import processpic as pp
import neuralNet


# note:
# 1. still finding a way to get the test data directory instand of putting
# them in a csv file and read again.
# 2. don't know why it is not woring for MLP

class categorize:
    # classNum = 41
    # width = 25
    # height = 37
    # channel = 1
    # sampleNum = 1140
    # resultFolder = 'result/'
    # testFolder = 'testBlank/'
    # trainFolder = 'train/'

    def __init__(self,
    model = "MLP",
    classNum=41,
    width=25,
    height=37,
    channel = 1,
    sampleNum = 1140,
    resultFolder = 'result/',
    testFolder = 'testBlank/',
    trainFolder = 'train/'
                 ):
        self.model = model
        self.width = width
        self.height = height
        self.resultFolder = resultFolder
        self.testFolder = testFolder
        self.trainFolder = trainFolder
        self.picmat = pp.picMat(width, height, channel,
                           classNum, sampleNum)

    def trainningProcess(self):

        self.picmat.saveRawMatrix()  # create x_train, y_label
        self.picmat.saveTrainY()  # create y_train
        self.picmat.saveTrainData()  # create train

        if (self.model == 'MLP'):
            x_train, y_train = neuralNet.load_TrainData(classNumber = self.classNumber)
            net = neuralNet.build_mlp()
        elif (self.model == 'CNN'):
            x_train, y_train = neuralNet.load_TrainData2D(classNumber = self.classNumber,
                                                          width=self.width, height=self.height)
            net = neuralNet.buildCNN()
        else:
            print 'ERROR: Please select a model #MLP or #CNN !'

        print "Start training:"
        net.fit(x_train, y_train)

        return net



    def createCategorizedFolders(self):
        print 'create category folders'
        if not os.path.exists(self.resultFolder):
            os.mkdir(self.resultFolder)
        categoryList = [x[0][len(self.trainFolder):] for x in os.walk(self.trainFolder)][1:]
        for cat in categoryList:
            if not os.path.exists(self.resultFolder + cat):
                os.mkdir(self.resultFolder + cat)
        print 'creating folders succeeded'
        return categoryList


    def categorizingProcess(self, net, categoryList):
        print 'start categorizing images'
        testData = os.listdir(self.testFolder)
        testData.sort()
        # os.system("find /Users/zeng/ubuntuSharedFolder/OCR -name '.DS_Store' -depth -exec rm {} \;") ##clear .DS_Store file in mac
        for root in testData:
            curPath = self.testFolder + root +'/nonBlank'+ '/'
            print 'curent path: ', curPath
            self.picmat.saveRawTestData(curPath)
            x_test, y_label = neuralNet.load_TestData2D(width=self.width, height=self.height)
            y_pred = net.predict(x_test)
            dataList = os.listdir(curPath)
            dataList.sort()
            for i in range(y_pred.shape[0]):
                 shutil.copy(curPath+dataList[i],
                             self.resultFolder + categoryList[y_pred[i].tolist().index(max(y_pred[i]))])
        print 'categorizing succeeded'
        return


    # categorizeForWhole is used for the testBlank Folder and will save result in the result folder
    # if deleteTestFolder is false, it will not delete the original testBlank Folder
    def categorizeForWhole(self, deleteTestFolder = False):
        net = self.trainningProcess()
        categoryList = self.createCategorizedFolders()
        self.categorizingProcess(net,categoryList)

        if deleteTestFolder == True:
            print 'delete original directory'
            shutil.rmtree(self.testFolder)
        return

    def getTrainingDataList(self):
        x_train = np.zeros((1, self.width * self.height))
        classes = os.listdir(self.trainFolder)
        classes.sort()
        for root in classes:
            for files in os.listdir(self.trainFolder + root + '/'):
                data = self.picmat.convertMatHelper(self.trainFolder + root+ '/' + files)
                x_train = np.concatenate((x_train, data), axis=0)
        return x_train

    def getTestFolderDataList(self, root):
        curPath = self.testFolder+root+'/' + 'nonBlank'
        x_test = np.zeros((1, self.width * self.height))
        classes = os.listdir(curPath)
        classes.sort()
        fileList = []
        for root in classes:
            for files in os.listdir(curPath):
                data = self.picmat.convertMatHelper(curPath + files)
                x_test = np.concatenate((x_test, data), axis=0)
                fileList.append(curPath+files)
        return x_test, fileList

    # deleteCategorizedFiles is used for deleting those files that are exist in the training folder
    # because they are already categorized
    def deleteCategorizedFiles(self):
        x_train = self.getDataList()

        x_test, fileList = getTestFolderDataList()
if __name__ == '__main__':
    c = categorize()
    # c.categorizeForWhole()
    print c.getDataList()