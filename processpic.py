'''
Created on Feb 29, 2016

@author: ly
'''
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os 
import os.path
import glob
import pandas as pd
from scipy import stats
from scipy.misc import toimage


#----------------this is py file to convert PNG file into matrix for future machine learning------
#----------------it will create x_train.csv and y_train.csv-------
##                                                               @author Tianle Zhang
#                                                                @email: tianlz1@uci.edu

class picMat:
    
    width     = 25
    height    = 37
    channel   = 1
    # classNum  = 41
    classNum = 52
    # sampleNum = 1140
    sampleNum = 1800
    # sample Num = train folder item num - classNum
    def _init_(self, width, height, channel, classNum, sampleNum):
        self.width = width        #the width of the matrix
        self.height = height      #the height of the matrix
        self.channel = channel    #the channel of the picture
        self.classNum = classNum
        self.sampleNum = sampleNum
    
    ####  convert one picture to 1-D matrix [width*height , 1]
    #      height width units is pixel
    #      didn't binary
    #      please use black and white picture
    
    def convertMatHelper(self, filename):
        
        img = Image.open(filename).convert('I')
        
        img = img.resize((self.width,self.height),Image.ANTIALIAS)  #resize
        data = np.array(img).reshape(self.width*self.height,1).T
        # for thresholding
        low_values_indices = data < 180
        data[low_values_indices] = 0
        high_values_indices = data >= 180
        data[high_values_indices] = 255
        # toimage(data).show()
        return data
    
    
    #### combine all pictures in one 'train' folder into a x_train matrix [number of pictures , pixels]
    #    Convert this matrix into pd.Dataframe
    #    save the dataframe into csv file 
    #    At same time , it will create a y_label csv for the x_train (one to one)
    
    def __combineTrainMatrix(self, foldername='train/'):
        classname = 0      # for class name
        label_y = []       # for y_train , label y
        x_train = np.zeros((1,self.width*self.height))
        
        #lst = os.listdir(foldername)
        #JZ modified because there is a txt file in the train folder and cause an error
        lst = [ele for ele in os.listdir(foldername) if os.path.isdir(foldername+ele)]
        
        lst.sort()
        
        # convert all the pictures into matrix and combine them into one matrix
        for root1 in lst: 
            classname += 1        
            for files in os.listdir(foldername+root1+'/'):
                
                data = self.convertMatHelper(foldername + root1 + '/' + files)
                x_train = np.concatenate((x_train,data),axis=0)
                label_y.append(classname)
        
        x_train = pd.DataFrame(x_train)
        x_train = x_train.drop(0,axis=0)
        
        label_y = pd.DataFrame(np.array(label_y),columns=['class'])

        return x_train, label_y
    
    # save the train_x and the train_y
    def saveRawMatrix(self, X_csvName='x_train.csv', Y_csvName='y_label.csv'):
        
        train_x,label_y = self.__combineTrainMatrix()
        train_x.to_csv(X_csvName)
        label_y.to_csv(Y_csvName)
    
    #### using y lable to create y_train.csv
    #    y is a vector of classnumber * 1
    #    y for one picture looks like [0,0,0,1,0,0,0,0] means it belongs to NO.4 class
    def saveTrainY(self, filename='y_train.csv',loadfile='y_label.csv'):
        
        try:
            label_Y = pd.read_csv(loadfile)
        
        except IOError:
            print "Error: can't find file or read data"
        
        else:
            label_Y = np.array(label_Y.iloc[:,1:]).T
            y = np.zeros((self.sampleNum,self.classNum))
            
            for i in range(self.sampleNum):
                y[i][label_Y[0][i]-1] = 1
            
            y = pd.DataFrame(y).to_csv(filename)
            
            return y
    
    #### combine x_train and y_train csv file into one file 
    #    And shuffle the order of the data for the future use
    #    The first column will be class
    #    Create a train.csv
     
    def saveTrainData(self, shuffle=True, normalize=True, label=False, filename='train.csv'):
        try:
            x_train = pd.read_csv('x_train.csv')
            
            if not label:
                y_train = pd.read_csv('y_train.csv')
            else:
                y_train = pd.read_csv('y_label.csv')
        
        except IOError:
            print "Error: cannot find x_train.csv or y_train.csv or y_label.csv, please use .saveRawMatrix() or .saveTrainY()"
        # make x_train and y_train values to [0,1]
        if normalize:
            x_train = x_train.iloc[:,1:]/255
            y_train = y_train.iloc[:,1:]
        
        # concatenate x_train and y_train
        all_train = pd.DataFrame(pd.concat([y_train, x_train],axis=1))
        
        # shuffle the order
        if shuffle:
            all_train = all_train.reindex(np.random.permutation(all_train.index))
            all_train.to_csv('train.csv')
        else:
            all_train.to_csv('train.csv')
            
    ## Load test pictures and transfer them into matrix
       ## create the x_test.csv
    # def __CombineTestData(self,foldername='test/'):
    #     currentClass = 'whatever'
    #     ClassName = 0
    #     label_y=[]
    #
    #     x_test = np.zeros((1,self.width*self.height))
    #
    #     lst = os.listdir(foldername)
    #     lst.sort()
    #     #print lst
    #     for files in lst:
    #
    #         if (files[:2] != currentClass):
    #             currentClass = files[:2]
    #             ClassName+=1
    #
    #         data = self.convertMatHelper(foldername+files)
    #         x_test = np.concatenate((x_test,data),axis=0)
    #         label_y.append(ClassName)
    #
    #     x_test = pd.DataFrame(x_test)
    #     x_test = x_test.drop(0,axis=0)
    #     #x_test.to_csv('x_test.csv')
    #
    #     label_y = pd.DataFrame(np.array(label_y),columns=['class'])
    #     #label_y.to_csv('y_testLabel.csv')
    #
    #     return x_test,label_y
    #
    def __CombineTestData(self, foldername='test/'):
        currentClass = 'whatever'
        ClassName = 0
        label_y=[]

        x_test = np.zeros((1,self.width*self.height))

        lst = os.listdir(foldername)
        lst.sort()
        print lst
        for filefolder in lst:
            for files in sorted(os.listdir(foldername+filefolder+"/")):
                # print files
                if (files[:2] != currentClass):
                    currentClass = files[:2]
                    ClassName+=1

                data = self.convertMatHelper(foldername+filefolder+"/" + files)
                x_test = np.concatenate((x_test,data),axis=0)
                label_y.append(ClassName)

        x_test = pd.DataFrame(x_test)
        x_test = x_test.drop(0,axis=0)
        #x_test.to_csv('x_test.csv')

        label_y = pd.DataFrame(np.array(label_y),columns=['class'])
        #label_y.to_csv('y_testLabel.csv')

        return x_test,label_y

    def getTestData(self, foldername ='test/'):
        currentClass = 'whatever'
        # ClassName = 0
        label_y = []

        x_test = np.zeros((1, self.width * self.height))

        lst = os.listdir(foldername)
        lst.sort()
        # print lst
        for files in lst:
            # print file
            data = self.convertMatHelper(foldername + files)
            x_test = np.concatenate((x_test, data), axis=0)
        # x_test = pd.DataFrame(x_test)
        # x_test = x_test.drop(0, axis=0)
        x_test /=255
        x_test = x_test.reshape(-1,1, self.height, self.width)
        return x_test




    def saveRawTestData(self,foldername, X_csvName='x_test.csv', Y_csvName='y_testLabel.csv'):
        x_test, label_y = self.__CombineTestData(foldername)
        x_test.to_csv(X_csvName)
        label_y.to_csv(Y_csvName)

           
if __name__ == '__main__': 
    
    p = picMat()
    #
    p.saveRawMatrix()    #create x_train, y_label
    p.saveTrainY()       #create y_train
    p.saveTrainData()    #create train
    p.saveRawTestData("test/")  #create x_test, y_testLabel
    # p.convertMatHelper("test/1/Ca1.png")
        

    
        
