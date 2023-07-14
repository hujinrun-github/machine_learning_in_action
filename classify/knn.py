from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqlDistances = sqDiffMat.sum(axis=1)
    distances = sqlDistances**0.5
    sortedDistIndices = distances.argsort()  # sort and return the indices
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(fileName, sampleLen, datalen, labelPosition=-1):
    with open(fileName, 'r') as f:
        arrayOnlines = f.readlines()
        numberOfLines = len(arrayOnlines)
        # print(numberOfLines)
        # print(sampleLen)
        returnMat = zeros((numberOfLines, sampleLen))
        classLabelVector = []
        index = 0

        for line in arrayOnlines:
            line = line.strip()
            listFromLine = line.split('\t')
            if len(listFromLine) != datalen:
                continue
            try:
                returnMat[index, :] = listFromLine[0:sampleLen]
            except Exception as e:
                print("ERROR index:{0},listFromLine:{1},lable:{2}".format(
                      index, listFromLine[0:sampleLen], listFromLine[labelPosition]))
                return
            else:
                # print("ERROR index:{0},istFromLine:{1},lable:{2}".format(
                # index, listFromLine[0:sampleLen], listFromLine[labelPosition]))
                
                classLabelVector.append(int(listFromLine[labelPosition]))
                index += 1
        returnMat = returnMat[:index, :]
        return returnMat, classLabelVector


def plotData(sampleData, labelMap, labelData,  xindex, yindex):
    fig, ax = plt.subplots()
    m = {1: 'o', 2: 's', 3: 'D', 4: '+'}
    cm = list(map(lambda x: m[x], labelData))
    ax.scatter(sampleData[:, xindex], sampleData[:,
               yindex], c=list(labelData), linewidths=3)
    ax.set_xlabel(labelMap[xindex])
    ax.set_ylabel(labelMap[yindex])


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet /= tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
