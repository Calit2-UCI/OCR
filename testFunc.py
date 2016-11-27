"""
@author: Jenny Zeng zhaohuaz@uci.edu
this is a file for removing boundaries of an image in a directory
"""

import platform
import numpy as np
import shutil
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.misc import toimage
import matplotlib.pyplot as plt
import os


def convertMatHelper(filename):
	"""
	open an image, make it a 2-d array and normalize to 0 and 1
	:param filename: image location
	:return: a 2-d array
	"""
	img = Image.open(filename).convert('I')

	# img = img.resize((self.width,self.height),Image.ANTIALIAS)  #resize
	data = np.array(img)  # .reshape(self.width*self.height,1).T
	# for thresholding and mormalize
	low_values_indices = data < 180
	data[low_values_indices] = 0
	high_values_indices = data >= 180
	data[high_values_indices] = 1
	# toimage(data).show()
	return data


def constructPointsArray(data):
	"""
	construct a array in which each list is a point representing
	the coordinates of the pixels that are black (0)
	:param data: 2-d array
	:return: a array in which each list is a point representing
	the coordinates of the pixels that are black (0)
	"""
	points = []
	for i in range(len(data)):
		for j in range(len(data[i])):
			if data[i][j] == 0:
				points.append([i, j])
	return np.array(points)


def findBoundary(data, points):
	"""
	use convex hull algorithm to find the boundary of the image
	and make the pixels that are used to construct the polygon in convex hull.
	:param data: 2 d array of image
	:param points: corresponding coordinates of the pixels that are black (0)
	:return: modified 2-d array
	"""
	hull = ConvexHull(points)
	plt.plot(points[:, 0], points[:, 1], 'o')
	for simplex in hull.simplices:
		plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
		p1 = points[simplex]
		p2 = points[simplex]

		data[p1[0][0]][p1[0][1]] = 1
		data[p1[1][0]][p1[1][1]] = 1
		data[p2[0][0]][p2[0][1]] = 1
		data[p2[1][0]][p2[1][1]] = 1
	### display the convexHull result
	# toimage(data).show()
	# plt.show()

	hull.close()
	return data


def processImg(filename, destination, toFormat=".jpg", times=25):
	"""
	remove boundary in a image for times
	:param filename: the image location
	:param destination: where the modified image will be stored
	:param toFormat: the result format of the image
	:param times: times of running removing boundary process
	:return: None
	"""
	data = convertMatHelper(destination + filename)
	i = 0
	while i < times:
		points = constructPointsArray(data)
		findBoundary(data, points)
		toimage(data).save(destination + filename + "-" + str(i) + toFormat)
		plt.savefig(destination + filename + "-plt-" + str(i) + toFormat)
		plt.close()
		i += 1


def main(originalFolder="./test/", resultFolder="./convertedImges/"):
	"""
	process all the images in the original folder and store them into the
	result folder.
	:param originalFolder: where original images from
	:param resultFolder: where the modified images will be stored
	:return: None
	"""
	if os.path.exists(resultFolder):
		shutil.rmtree(resultFolder)
	for subDir in os.listdir(originalFolder):
		for img in os.listdir(originalFolder + subDir):
			destination = resultFolder + subDir + "/" + img + "-dir/"
			print "current destination: ", destination
			print "current image: ", img
			if not os.path.exists(destination):
				os.makedirs(destination)
			shutil.copy(originalFolder + "/" + subDir + "/" + img, destination)
			processImg(img, destination)


if __name__ == '__main__':
	# to remove the .DS_Store file on mac
	if (platform.system() == "Darwin"):
		os.system("find . -name '.DS_Store' -type f -delete")
	main()
