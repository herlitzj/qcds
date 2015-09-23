import matplotlib
matplotlib.use("Agg")
import urllib2
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plt

target_url = ("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
data = urllib2.urlopen(target_url)

xList = []
labels = []
names = []
firstLine = True
for line in data:
	if firstLine:
		names = line.strip().split(";")
		firstLine = False
	else:
		row = line.strip().split(";")	
		labels.append(float(row[-1]))
		row.pop()
		floatRow = [float(num) for num in row]
		xList.append(floatRow)

indices = range(len(xList))
xListTest = [xList[i] for i in indices if i%3 == 0]
xListTrain = [xList[i] for i in indices if i%3 !=0]
labelsTest = [labels[i] for i in indices if i%3 == 0]
labelsTrain = [labels[i] for i in indices if i%3 !=0]

xTrain = numpy.array(xListTrain)
yTrain = numpy.array(labelsTrain)
xTest = numpy.array(xListTest)
yTest = numpy.array(labelsTest)

alphaList = [0.1**i for i in [0,1,2,3,4,5,6]]

rmsError = []
for alph in alphaList:
	wineRidgeModel = linear_model.Ridge(alpha=alph)
	wineRidgeModel.fit(xTrain, yTrain)
	rmsError.append(numpy.linalg.norm((yTest-wineRidgeModel.predict(xTest)), 2)/sqrt(len(yTest)))

print("RMS Error	alpha")
for i in range(len(rmsError)):
	print(rmsError[i], alphaList[i])

x = range(len(rmsError))
plt.plot(x, rmsError, 'k')
plt.xlabel('-log(alpha)')
plt.ylabel('Error (RMS)')
plt.savefig('ridgeWine.jpg')

	
