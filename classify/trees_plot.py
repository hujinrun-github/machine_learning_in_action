import matplotlib.pyplot as plt
from . import trees

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    thisDepth = 0
    for key in secondDict.keys():
        if type(secondDict.keys()).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth += 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


"""
Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
Returns:
    无
"""
# 函数说明:绘制结点


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    arrow_args = dict(arrowstyle="<-")  # 定义箭头格式
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)    #设置中文字体
    # print("nodeTxt:{}, centerPtr:{}, parentPt:{}, nodeType:{}".format(nodeTxt, centerPt, parentPt, nodeType))
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',  # 绘制结点
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


"""
Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注的内容
Returns:
    无
"""
# 函数说明:标注有向边属性值


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]  # 计算标注位置
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center",
                        ha="center", rotation=30)


"""
Parameters:
    myTree - 决策树(字典)-
    parentPt - 标注的内容
    nodeTxt - 结点名
Returns:
    无
"""
# 函数说明:绘制决策树


def plotTree(myTree, parentPt, nodeTxt):
    # decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # 设置结点格式
    # leafNode = dict(boxstyle="round4", fc="0.8")  # 设置叶结点格式
    numLeafs = getNumLeafs(myTree)  # 获取决策树叶结点数目，
    # 决定了树的宽度
    # depth = getTreeDepth(myTree)  # 获取决策树层数
    firstStr = next(iter(myTree))  # 下个字典
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0/plotTree.totalW, plotTree.yOff)  # 中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标注有向边属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)  # 绘制结点
    secondDict = myTree[firstStr]  # 下一个字典，
    # 也就是继续绘制子结点
    plotTree.yOff = plotTree.yOff - 0.5/plotTree.totalD  # y偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':  # 测试该结点是否为字典，
            # 如果不是字典，
            # 代表此结点为叶子结点
            plotTree(secondDict[key], cntrPt, str(key))  # 不是叶结点，
            # 递归调用继续绘制
        else:  # 如果是叶结点，绘制叶结点，
            # 并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff,
                     plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 0.5/plotTree.totalD


"""
Parameters:
    inTree - 决策树(字典)
Returns:
    无
"""
# 函数说明:创建绘制面板


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white', figsize=(10, 10))  # 创建fig
    fig.clf()  # 清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))  # 获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))  # 获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW  # w = 3, xOff = -0.5/3
    plotTree.yOff = 1.0  # y偏移
    plotTree(inTree, (0.5, 1.0), '')  # 绘制决策树
    plt.show()  # 显示绘制结果


if __name__ == '__main__':
    dataSet, labels = trees.createDataSet()
    featLabels = []
    myTree = trees.createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
