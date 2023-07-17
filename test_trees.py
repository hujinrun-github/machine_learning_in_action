from classify import trees
from classify import trees_plot
from classify import tree_apply
import matplotlib.pyplot as plt
from numpy import *

# myDat, labels = trees.createDatSet()
# print(myDat)
# # print(trees.calcShannonEng(myDat))
# # print(trees.splitDataSet(myDat, 0, 1))
# # print(trees.chooseBestFeatureToSplit(myDat))
# # print(trees.calcShannonEng(trees.splitDataSet(myDat, 1, 1)))
# # print(trees.calcShannonEng(trees.splitDataSet(myDat, 1, 0)))
# print(trees.majorityCnt([1, 2, 3, 1, 1, 1]))
# print(trees.createTree(myDat, labels))
# dataSet, labels = trees.createDataSet()
# featLabels = []
# myTree = trees.createTree(dataSet, labels, featLabels)
# print(myTree)
# print(featLabels)
# trees_plot.createPlot(myTree)
tree_apply.classifyHomeData("./dataSet/home-data-for-ml-course/train.csv")