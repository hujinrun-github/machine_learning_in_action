from numpy import *
from os import listdir
from . import knn


def img2vector(fileName, lineNum=32, columnNum=32):
    returnVect = zeros((1, 1024))
    with open(fileName) as fr:
        for i in range(lineNum):
            lineStr = fr.readline()
            for j in range(columnNum):
                returnVect[0, columnNum*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(trainFilePath="./dataSet/knn_image/trainingDigits", testFilePath="./dataSet/knn_image/testDigits"):
    hwLabels = []
    trainingFileList = listdir(trainFilePath)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('%s/%s' % (trainFilePath, fileNameStr))
    testFileList = listdir(testFilePath)
    errCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("%s/%s" % (testFilePath, fileNameStr))
        classifierResult = knn.classify0(
            vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with:%d, the real answer is:%d" %
              (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errCount += 1.0
    print("\nthe total number of error is:%d" % errCount)
    print("\nthe total error rate is:%f" % (errCount/float(mTest)))
