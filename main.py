import math
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial


def euclidean(x, y):
    return math.sqrt(sum([((x_i - y_i) ** 2) for x_i, y_i in zip(x, y)]))


def manhattan(x, y):
    return sum([abs(x_i - y_i) for x_i, y_i in zip(x, y)])


def chebyshev(x, y):
    return max([abs(x_i - y_i) for x_i, y_i in zip(x, y)])


def getDistFunctionTo(name, to):
    if name == "manhattan":
        return partial(manhattan, y=to)
    elif name == "euclidean":
        return partial(euclidean, y=to)
    else:
        return partial(chebyshev, y=to)


def uniform(x):
    if abs(x) < 1:
        return 1 / 2
    return 0


def triangular(x):
    if abs(x) < 1:
        return 1 - abs(x)
    return 0


def epanechnikov(x):
    if abs(x) < 1:
        return (3 / 4) * (1 - (x ** 2))
    return 0


def quartic(x):
    if abs(x) < 1:
        return (15 / 16) * ((1 - x ** 2) ** 2)
    return 0


def getKernelFunction(name):
    if name == "uniform":
        return uniform
    elif name == "triangular":
        return triangular
    elif name == "epanechnikov":
        return epanechnikov
    else:  # quartic
        return quartic


df = pd.read_csv("data/normies.csv", sep=',')
dataSize = len(df.axes[0])
neighborhoodSize = round(math.sqrt(dataSize))


def getListOfWindowWidth(dataFrame, distFun, winType):
    if winType == "variable":
        return [i for i in range(1, neighborhoodSize)]
    tmpList = dataFrame.values.tolist()
    f = getDistFunctionTo(distFun, [0] * (len(dataFrame.iloc[0]) - 3))
    sortedList = sorted(tmpList, key=lambda x: f(x[:-3]))
    maxDist = f(sortedList[-1]) - f(sortedList[0])
    widthList = []
    a = maxDist / neighborhoodSize
    z = a
    for i in range(round(a), round(neighborhoodSize)):
        widthList.append(z)
        z += a
    return widthList


def getClass(obj):
    tmp = obj[-3:]
    if tmp == [1, 0, 0]:
        return 1
    elif tmp == [0, 1, 0]:
        return 2
    elif tmp == [0, 0, 1]:
        return 3
    else:
        return 4


def getRegression(tObj, trainSet, dFunc, kernelFunc, winType, div, clazz):
    targetValues = []
    for elem in trainSet:
        if getClass(elem) == clazz:
            targetValues.append(1)
        else:
            targetValues.append(0)

    distFunc = getDistFunctionTo(dFunc, tObj[:-3])
    kernelF = getKernelFunction(kernelFunc)

    result = sum([targetValues[k] for k in range(0, len(trainSet))]) / len(trainSet)

    if winType != "fixed":
        div = distFunc(trainSet[div][:-3])

    if div != 0:
        h1 = sum([targetValues[k] * kernelF(distFunc(trainSet[k][:-3]) / div) for k in range(0, len(trainSet))])
        h2 = sum([kernelF(distFunc(trainSet[k][:-3]) / div) for k in range(0, len(trainSet))])
        if h2 != 0:
            result = h1 / h2
    else:
        if distFunc(trainSet[0][:-3]) == 0:
            j = 0
            while distFunc(trainSet[j][:-3]) == 0 and j < len(trainSet):
                j += 1
            n = [targetValues[k][:-3] for k in range(0, j + 1)]
            if len(n) != 0:
                result = sum(n) / len(n)
    return result


def setConfusionMatrix(confMatrix, tObj, trainSet, dFunc, kernelFunc, winType, div):
    maxRegression = -1
    predictedClass = -1

    for clazz in range(1, 4):  # let tObj is type classes
        tmpTrainSet = trainSet
        regression = getRegression(tObj, tmpTrainSet, dFunc, kernelFunc, winType, div, clazz)
        if regression > maxRegression:
            maxRegression = regression
            predictedClass = clazz

    trueClazz = getClass(tObj)
    confMatrix[trueClazz - 1][predictedClass - 1] += 1

    return 0


def getFMeasure(confMatrix):
    n = 3

    def getHorizontalSum(line):
        x = 0
        for j in range(n):
            x += confMatrix[line][j]
        return x

    def getVerticalSum(column):
        x = 0
        for j in range(n):
            x += confMatrix[j][column]
        return x

    All = 0
    for i in range(n):
        for j in range(n):
            All += confMatrix[i][j]

    tp = [confMatrix[i][i] for i in range(n)]
    c = [getHorizontalSum(i) for i in range(n)]
    p = [getVerticalSum(i) for i in range(n)]
    fp = [c[i] - tp[i] for i in range(n)]
    fn = [p[i] - tp[i] for i in range(n)]
    recall = [tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) != 0 else 0 for i in range(n)]
    precision = [tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) != 0 else 0 for i in range(n)]

    precisionW = 0
    for i in range(n):
        if p[i] != 0:
            precisionW += c[i] * confMatrix[i][i] / p[i]

    precisionW /= All

    recallW = 0
    for i in range(n):
        recallW += confMatrix[i][i]
    recallW /= All

    f1s = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i in
           range(n)]

    microF = 0
    for i in range(n):
        microF += c[i] * f1s[i] / All

    return microF


windowTypes = ["fixed", "variable"]
distFunctions = ["euclidean", "manhattan", "chebyshev"]
kernelFunctions = ["uniform", "triangular", "epanechnikov", "quartic"]

maxFMeasure = -1
bestDistFunction = None
bestKernelFunction = None
bestWindowType = None
bestWindowParam = None
bestArguments = []

for windowType in windowTypes:
    for distFunction in distFunctions:
        for kernelFunction in kernelFunctions:
            dividersList = getListOfWindowWidth(df, distFunction, windowType)
            for divider in dividersList:
                confusionMatrix = [[0 for i in range(0, 3)] for j in range(0, 3)]
                for pos in range(0, dataSize):
                    dfList = df.values.tolist()
                    targetPoint = dfList[pos]
                    del dfList[pos]
                    distF = getDistFunctionTo(distFunction, targetPoint[:-3])
                    sortedDfList = sorted(dfList, key=lambda x: distF(x[:-3]))
                    setConfusionMatrix(confusionMatrix, targetPoint, sortedDfList, distFunction, kernelFunction,
                                       windowType, divider)
                fMeasure = getFMeasure(confusionMatrix)

                if fMeasure > maxFMeasure:
                    maxFMeasure = fMeasure
                    bestDistFunction = distFunction
                    bestKernelFunction = kernelFunction
                    bestWindowType = windowType
                    bestWindowParam = divider
                    bestArguments = dividersList

bestFMeasureValue = []
confusionMatrix = [[0 for i in range(0, 3)] for j in range(0, 3)]

for divider in bestArguments:
    confusionMatrix = [[0 for i in range(0, 3)] for j in range(0, 3)]
    for pos in range(0, dataSize):
        dfList = df.values.tolist()
        targetPoint = dfList[pos]
        del dfList[pos]
        distF = getDistFunctionTo(bestDistFunction, targetPoint[:-3])
        sortedDfList = sorted(dfList, key=lambda x: distF(x[:-3]))
        setConfusionMatrix(confusionMatrix, targetPoint, sortedDfList, bestDistFunction, bestKernelFunction,
                           bestWindowType, divider)
    bestFMeasureValue.append(getFMeasure(confusionMatrix))

file = open("save.txt", "w")
file.write(str(maxFMeasure) + '\n')
file.write(bestDistFunction + '\n')
file.write(bestWindowType + '\n')
file.write(str(bestWindowParam) + '\n')
file.write(bestKernelFunction + '\n')
file.close()

print("Best hyperparameters\nF-measure: " + str(maxFMeasure) + "\ndistant function: " + bestDistFunction +
      "\nwindow type: " + bestWindowType + "\nwindow param: " + str(bestWindowParam) + "\nkernel function: "
      + bestKernelFunction)

fig, ax = plt.subplots()
ax.plot(bestArguments, bestFMeasureValue)
plt.show()
fig.savefig('function graph')
