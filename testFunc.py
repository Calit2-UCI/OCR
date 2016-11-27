import platform

import numpy as np
import shutil
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.misc import toimage
import matplotlib.pyplot as plt
import os

def convertMatHelper( filename):
    img = Image.open(filename).convert('I')

    # img = img.resize((self.width,self.height),Image.ANTIALIAS)  #resize
    data = np.array(img)  # .reshape(self.width*self.height,1).T
    # for thresholding
    low_values_indices = data < 180
    data[low_values_indices] = 0
    high_values_indices = data >= 180
    data[high_values_indices] = 1
    # toimage(data).show()
    return data

def constructPointsArray(data):
    points = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == 0:
                points.append([i, j])
    return np.array(points)


def findBoundary(data, points):
    hull = ConvexHull(points)
    plt.plot(points[:, 0], points[:, 1], 'o')
    # print "attention: ", len(data[0])
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        p1 = points[simplex]
        p2 = points[simplex]

        data[p1[0][0]][p1[0][1]]= 1
        data[p1[1][0]][p1[1][1]]= 1
        data[p2[0][0]][p2[0][1]] = 1
        data[p2[1][0]][p2[1][1]]= 1

    # toimage(data).show()
    # plt.show()

    hull.close()
    return data

def processImg(filename, destination, toFormat= ".jpg", ):
    # destination = resultFolder+ fileDirectory + filename +"-dir/"
    # if not os.path.exists(destination):
    #     os.makedirs(destination)
    data = convertMatHelper(destination + filename)
    # area = (len(data)-1) * (len(data[0])-1)
    # print ("my area: ", area)
    i = 0
    while i<25:
        points = constructPointsArray(data)
        data= findBoundary(data, points)
        toimage(data).save(destination + filename+"-" + str(i) + toFormat)
        plt.savefig(destination + filename+"-plt-"+str(i)+ toFormat)
        # toimage(data).save(resultFolder + fileDirectory + filename+"-" + str(i) + toFormat)
        # plt.savefig(resultFolder+fileDirectory+filename+"-plt-"+str(i)+ toFormat)
        plt.close()
        i+=1


def main(originalFolder = "./test/",resultFolder = "./convertedImges/"):
    if os.path.exists(resultFolder):
        shutil.rmtree(resultFolder)
    for subDir in os.listdir(originalFolder):
        for img in os.listdir(originalFolder+subDir):
            destination = resultFolder + subDir +"/"+ img + "-dir/"
            print destination
            print img
            if not os.path.exists(destination):
                os.makedirs(destination)
            shutil.copy(originalFolder +"/"+ subDir + "/" + img, destination)
            processImg( img, destination)



if __name__ == '__main__':
    if(platform.system()== "Darwin"):
        os.system("find . -name '.DS_Store' -type f -delete")
    # data = convertMatHelper("test/3/Nm.png")
    # print data
    # print len(data)
    # print data
    # points = np.random.rand(30, 2)  # 30 random points in 2-D

    # print points
    # a = constructPointsArray(data)
    # print "a: ", a
    # print "a end"

    # findBoundary(data, a)
    # process("Po1")
    main()