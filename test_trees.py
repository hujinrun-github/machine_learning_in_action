from classify import trees
import matplotlib.pyplot as plt
from numpy import *

myDat, labels = trees.createDatSet()
print(myDat)
# print(trees.calcShannonEng(myDat))
# print(trees.splitDataSet(myDat, 0, 1))
# print(trees.chooseBestFeatureToSplit(myDat))
# print(trees.calcShannonEng(trees.splitDataSet(myDat, 1, 1)))
# print(trees.calcShannonEng(trees.splitDataSet(myDat, 1, 0)))
print(trees.majorityCnt([1, 2, 3, 1, 1, 1]))
print(trees.createTree(myDat, labels))