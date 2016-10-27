import os
import shutil
import re


def checkPrecision(resultFolder='result/'):
	precisionList = []
	for dataFolder in sorted(os.listdir(resultFolder)):  # Bg
		curPath = resultFolder + dataFolder  # result/Bg
		dataSet = os.listdir(curPath)
		correct = 0
		total = len(dataSet)
		for data in dataSet:
			if data[:2] == dataFolder:
				correct += 1
		precision = correct / total * 100
		precisionList.append((dataFolder, precision))
		print "Folder: ", dataFolder, "precision: ", precision


def dataFilter(testDataFolder="dataFolder/", trainingDataFolder="train/",
			   groupname="Folder184.1-Korean-"):
	"""
	exclude data in training data set from test data set.
	"""
	hashTable = {}
	for dataFolder in os.listdir(testDataFolder):
		tag = re.search(".*-.*-(.*-.*)", dataFolder).group(1)
		hashTable.update({tag: os.listdir(testDataFolder + dataFolder)})
	for dataFolder in os.listdir(trainingDataFolder):
		for data in os.listdir(trainingDataFolder + dataFolder):
			dtag = re.search(".*-.*-(.*-.*)-.*", data).group(1)
			if hashTable.has_key(dtag) and data in hashTable.get(dtag):
				dataPath = testDataFolder + groupname + dtag + "/" + data
				os.remove(dataPath)
				print "data: " + dataPath + " is removed"


def getDataOutFromTestBlank(mainFolder="testBlank/", subFolder="nonBlank/", resultFolder="dataFolder/"):
	"""
	take data in nonblank folder in each subfolder of mainFolder out.
	"""
	## create new folder for result
	if os.path.exists(resultFolder):
		shutil.rmtree(resultFolder)
	os.mkdir(resultFolder)

	for dataFolder in os.listdir(mainFolder):
		curPath = mainFolder + dataFolder + "/" + subFolder
		desPath = resultFolder + dataFolder
		os.mkdir(desPath)
		for data in os.listdir(curPath):
			shutil.copy(curPath + data, desPath)
	return


if __name__ == "__main__":
	os.system("find . -name '.DS_Store' -type f -delete")
	# getDataOutFromTestBlank()
	dataFilter()
