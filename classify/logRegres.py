import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet(fileName:str):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(intX:list):
    return 1.0/(1+np.exp(-intX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose() # m*1
    m, n = np.shape(dataMatrix) # m*n
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) # m*1
        error = (labelMat - h) # m*1
        weights += alpha*dataMatrix.transpose()*error #(n*m)*(m*1)
    return weights

def stocGradAcent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights)) #scalar
        error = classLabels[i] - h
        weights = weights+ alpha * error * dataMatrix[i]
    return weights

def stocGradAcent1(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+i+j) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights += alpha*error*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def plotBestFit(wei):
    dataMat, labelMat = loadDataSet("./dataSet/log_grad/testSet.txt")
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c= 'green')
    x = np.arange(-3.0, 3.0,0.1)
    y = (-wei[0]- wei[1]*x)/wei[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def classifyVector(intX, weight):
    prob = sigmoid(sum(intX*weight))
    if prob > 0.5: return 1.0
    return 0.0

def colicTest():
    trainingSet = []
    trainingLabels = []
    with open("./dataSet/log_grad/horseColicTraining.txt") as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split("\t")
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAcent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    with open("./dataSet/log_grad/horseColicTest.txt") as frTest:
        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split("\t")
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is:%f"%(errorRate))
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is:%f"%(numTests, errorSum/float(numTests)))


def testGradAscent():
    dataArr, labelMat = loadDataSet("./dataSet/log_grad/testSet.txt")
    # weight = stogradAscent(dataArr, labelMat)
    weight = stocGradAcent1(np.array(dataArr), labelMat)
    print(weight)
    plotBestFit(weight)
    # print(weight)