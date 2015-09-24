import matplotlib
matplotlib.use ("Agg")
import urllib2
import csv
import numpy
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV
from math import sqrt
import matplotlib.pyplot as plot

#import data from csv file
data = csv.reader(open('../data/winequality-red.csv', 'r'))

#read data into list of lists
xList = []
labels = []
names = []
firstLine = True
for line in data:
	if firstLine:
		names = data.next()[0].split(';')
		firstLine = False
	else:
		row = line[0].split(';')
		labels.append(float(row[-1]))
		row.pop()
		floatRow = [float(num) for num in row]
		xList.append(floatRow)

#Normalize columns in x and labels
nrows = len(xList)
ncols = len(xList[0])

#calculate means and variances
xMeans = []
xSD = []
for i in range(ncols):
	col = [xList[j][i] for j in range(nrows)]
	mean = sum(col)/nrows
	xMeans.append(mean)
	colDiff = [(xList[j][i] - mean) for j in range(nrows)]
	sumSq = sum([colDiff[i] * colDiff[i] for i in range(nrows)])
	stdDev = sqrt(sumSq/nrows)
	xSD.append(stdDev)

#use mean and stddev to normalize xList
xNormalized = []
for i in range(nrows):
	rowNormalized = [(xList[i][j] - xMeans[j]) / xSD[j] for j in range(ncols)]
	xNormalized.append(rowNormalized)

#normalize labels
meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)]) / nrows)
labelNormalized = [(labels[i] - meanLabel) / sdLabel for i in range(nrows)]

#convert list of lists to np array for sklearn
yNormed = False
xNormed = False

if yNormed:
	Y = numpy.array(labelNormalized)
else:
	Y = numpy.array(labels)

if xNormed:
	X = numpy.array(xNormalized)
else:
	X = numpy.array(xList)

#Call LassoCV from sklearn
wineModel = LassoCV(cv=10).fit(X, Y)

plot.figure()
plot.plot(wineModel.alphas_, wineModel.mse_path_, ':')
plot.plot(wineModel.alphas_, wineModel.mse_path_.mean(axis=-1), label='Average MSE Across Folds', linewidth=2)
plot.axvline(wineModel.alpha_, linestyle='--', label = 'CV Estimate of Best alpha')
plot.semilogx()
plot.legend()
ax = plot.gca()
ax.invert_xaxis()
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

plot.xlabel('alpha')
plot.ylabel('Mean Square Error')
plot.axis('tight')
plot.savefig('./img/winelassoXYunnorm.jpg')

#print the value of alpha that minimizes CV error
print('alpha value that minimized CV error: ', wineModel.alpha_)
print('Minimum MSE: ', min(wineModel.mse_path_.mean(axis=-1)))

