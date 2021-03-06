import matplotlib
matplotlib.use ("Agg")
import csv
import numpy
from sklearn import datasets, linear_model
from math import sqrt
import matplotlib.pyplot as plot

dataOpen = open('./data/coupon_list_train.csv', 'r')
data = csv.reader(dataOpen)

xList = []
labels = []
names = []
firstLine = True

for line in data:
	if firstLine:
		names = data.next()[0].split(";")
		firstLine = False
	else:
		row = line[0].split(";")
		labels.append(float(row[-1]))
		row.pop()
		floatRow = [float(num) for num in row]
		xList.append(floatRow)

nrows = len(xList)
ncols = len(xList[0])

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

xNormalized = []
for i in range(nrows):
	rowNormalized = [(xList[i][j] - xMeans[j]) / xSD[j] for j in range(ncols)]
	xNormalized.append(rowNormalized)

meanLabel = sum(labels)/nrows
sdLabel = sqrt(sum([(labels[i] - meanLabel) * (labels[i] - meanLabel) for i in range(nrows)])/nrows)
labelNormalized = [(labels[i] - meanLabel)/sdLabel for i in range(nrows)]

beta = [0,0] * ncols

betaMat = []
betaMat.append(list(beta))

nSteps = 350
stepSize = 0.004

for i in range(nSteps):
	residuals = [0,0] * nrows
	for j in range(nrows):
		labelsHat = sum([xNormalized[j][k] * beta[k] for k in range(ncols)])
		residuals[j] = labelNormalized[j] - labelsHat
	corr = [0,0] * ncols
	for j in range(ncols):
		corr[j] = sum([xNormalized[k][j] * residuals[k] for k in range(nrows)]) / nrows
	iStar = 0
	corrStar = corr[0]
	for j in range(1, ncols):
		if abs(corrStar) < abs(corr[j]):
			iStar = j; corrStar = corr[j]
	beta[iStar] += stepSize * corrStar / abs(corrStar)
	betaMat.append(list(beta))

for i in range(ncols):
	coefCurve = [betaMat[k][i] for k in range(nSteps)]
	xaxis = range(nSteps)
	plot.plot(xaxis, coefCurve)

plot.xlabel("Steps Taken")
plot.ylabel(("Coefficient Values"))
plot.savefig("larsWine.jpg")







