import numpy as np
import math as mt
import random
import re
'''
Parameters:
    无
Returns:
    postingList - 实验样本切分的词条
    classVec - 类别标签向量
'''
# 函数说明:创建实验样本
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],       #切分的词条
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]#类别标签向量，1代表侮辱性词汇，0代表不是
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# 统计每个句子中，指定的单词是否出现
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in inputSet:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word:%s is not in my Vocabulary"%word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom) #在侮辱性句子中，某词出现的概率
    p0Vect = np.log(p0Num/p0Denom) #在非侮辱性句子中，某词出现的概率
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec) + np.log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print("trainMat", trainMat)
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    # print("P0", p0V)
    # print("P1", p1V)
    print(testEntry,"classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,"classified as:", classifyNB(thisDoc, p0V, p1V, pAb))

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def textParse(bigString):
    listOfTokens = re.split('\\W{2,}', bigString)
    result = []
    for res in listOfTokens:
        # print(res.split(" "))
        # print([tok.lower for tok in res.split(" ") if len(tok) >= 2])
        result.extend([tok.lower() for tok in res.split(" ") if len(tok) >= 2])
    return result

def spamTest():
    maxSize = 25
    docList = []
    classList = []
    fullText = []
    for i in range(0,maxSize):
        # print(i)
        with open('./dataSet/email/spam/%d.txt' % (i+1), encoding="ISO-8859-1") as f:
            wordList = textParse(f.read())
            # print(wordList)
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            # f.close()
        with open('./dataSet/email/ham/%d.txt' % (i+1), encoding="ISO-8859-1") as f:
            wordList = textParse(f.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
            # f.close()
    # print(docList)
    vocabList = createVocabList(docList)
    # print(vocabList)
    trainingSet = list(range(maxSize*2))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses =[]
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),  p0V, p1V, pSpam) != classList[docIndex]:
            errCount += 1
    print("the error rate is:", float(errCount)/len(testSet))
