import numpy as np
from classify import bayes

def test_simple_bayes():
    listOPosts, listClasses = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(listOPosts) #获取所有不重复的单词
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
    print(pAb)

# bayes.testingNB()
bayes.spamTest()
# bayes.textParse("dfa dfasdffa dfasdf dfasdf")
# test_simple_bayes()