from classify import knn
import matplotlib.pyplot as plt
from numpy import *

group, labels = knn.createDataSet()
# print(knn.classify0([0,0], group, labels,3))
datingDataMat, datingLabels = knn.file2matrix("./dataSet/datingTestSet2.txt",3, 4)
labelMap = {}
labelMap[0] = "fligt miles per year"
labelMap[1] = "the proportion of playing game"
labelMap[2] = "the liter of ice cream per week"
normDataSet, ranges, minVals = knn.autoNorm(datingDataMat)
print(normDataSet[:10])
# knn.plotData(datingDataMat, labelMap,datingLabels, 0,1)
# knn.plotData(datingDataMat, labelMap,datingLabels, 0,2)
# knn.plotData(datingDataMat, labelMap,datingLabels, 1,2)
# plt.show()
# print(datingDataMat[:10])
# print(datingLabels[:10])
# print(group)
# print(labels)