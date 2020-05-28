import xlrd
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 从excel中读取训练数据
def createDataSet(filename):
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)
    dataArr = []
    labelArr = []
    for i in range(worksheet.nrows):
        rowVals = np.array(worksheet.row_values(i))
        dataArr.append(rowVals[:-1])
        labelArr.append(rowVals[-1])


    return np.array(dataArr), np.array(labelArr)

# 在选中第i个alpha的基础上，随机选择j，当j与i相等时，继续随机，直到j不等于i则返回j
def selectJrand(i, m):
    j = i
    while (j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def smoSimple(dataArr, laberArr, C, toler, maxIter):
    K = 10
    '''
    for k in range(K):
        print("Doing fold ", k)
        trainDataArr = [x for i, x in enumerate(dataArr) if i % K != k]
        trainLabelArr = [x for i, x in enumerate(labelArr) if i % K != k]
        testDataArr = [x for i, x in enumerate(dataArr) if i % K == k]
        testLabelArr = [x for i, x in enumerate(labelArr) if i % K == k]
        trainDataArr = np.array(trainDataArr)
        trainLabelArr = np.array(trainLabelArr)
        testDataArr = np.array(testDataArr)
        testLabelArr = np.array(testLabelArr)
    '''
    '''
    k = 0
    print("Doing fold ", k)
    trainDataArr = [x for i, x in enumerate(dataArr) if i % K != k]
    trainLabelArr = [x for i, x in enumerate(labelArr) if i % K != k]
    testDataArr = [x for i, x in enumerate(dataArr) if i % K == k]
    testLabelArr = [x for i, x in enumerate(labelArr) if i % K == k]
    trainDataArr = np.array(trainDataArr)
    trainLabelArr = np.array(trainLabelArr).reshape(-1,1)
    testDataArr = np.array(testDataArr)
    testLabelArr = np.array(testLabelArr).reshape(-1,1)
    '''

    trainDataArr = dataArr
    trainLabelArr = labelArr.reshape(-1,1)

    m, n = np.shape(trainDataArr)
    alphas = np.zeros((m,1))            # 初始化alpha都为0，这样就会满足\sum(yi*\alpha_i) == 0
    b = 0
    for iter in range(maxIter):         # 循环maxIter次更新
        alphaPairsChanged = 0
        # 这段解释摘自https://github.com/apachecn/AiLearning/blob/master/src/py2.x/ml/6.SVM/svm-simple.py
        # 约束条件 (KKT条件是解决最优化问题的时用到的一种方法。我们这里提到的最优化问题通常是指对于给定的某一函数，求其在指定作用域上的全局最小值)
        # 0<=alphas[i]<=C，但由于0和C是边界值，我们无法进行优化，因为需要增加一个alphas和降低一个alphas。
        # 表示发生错误的概率: labelMat[i]*Ei 如果超出了 toler， 才需要优化。至于正负号，我们考虑绝对值就对了。
        # labelMat[i] = +1/-1，所以可以直接判断Ei是否超出toler
        '''
        # 检验训练样本(xi, yi)是否满足KKT条件
        yi*f(i) >= 1 and alpha = 0 (outside the boundary)
        yi*f(i) == 1 and 0<alpha< C (on the boundary)
        yi*f(i) <= 1 and alpha = C (between the boundary)
        '''
        alphaPairsChanged = 0
        for i in range(m):
            a = alphas * trainLabelArr
            w = sum(alphas * trainLabelArr * trainDataArr)
            fXi = float( np.dot(w, trainDataArr[i].transpose()) ) + b
            Ei = fXi - float(trainLabelArr[i])
            if ( (trainLabelArr[i] * Ei < -toler) and (alphas[i] < C)) or \
               ( (trainLabelArr[i] * Ei > toler) and (alphas[i] > 0) ):
                j = selectJrand(i, m)
                fXj = float( np.dot(w, trainDataArr[j].transpose()) ) + b
                Ej = fXj - float(trainLabelArr[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (trainLabelArr[i] != trainLabelArr[j]):
                    L = max(0, alphaJold - alphaIold)
                    H = min(C, C + alphaJold - alphaIold)
                else:
                    L = max(0, alphaIold + alphaJold - C)
                    H = min(C, alphaIold + alphaJold)
                if L == H:
                    print("L==H")
                    continue
                eta = 2.0 * np.dot(trainDataArr[i], trainDataArr[j].transpose()) \
                        - np.dot(trainDataArr[i], trainDataArr[i].transpose()) \
                        - np.dot(trainDataArr[j], trainDataArr[j].transpose())
                if eta >= 0:
                    print("eta >= 0")
                    continue
                alphas[j] -= trainLabelArr[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if ( abs(alphas[j] - alphaJold) < 1e-5):
                    print("j not moving enough")
                    alphas[j] = alphaJold
                    continue
                alphas[i] += trainLabelArr[j] * trainLabelArr[i] * \
                             (alphaJold - alphas[j])
                b1 = b - Ei - trainLabelArr[i] * (alphas[i] - alphaIold) * \
                    np.dot(trainDataArr[i,:], trainDataArr[i,:].transpose()) \
                    - trainLabelArr[j] * (alphas[j] - alphaJold) * \
                    np.dot(trainDataArr[j, :], trainDataArr[i, :].transpose())
                b2 = b - Ej - trainLabelArr[i] * (alphas[i] - alphaIold) * \
                     np.dot(trainDataArr[i,:], trainDataArr[j,:].transpose()) \
                     - trainLabelArr[j] * (alphas[j] - alphaJold) * \
                     np.dot(trainDataArr[j, :], trainDataArr[j, :].transpose())
                if (0 < alphas[i]) and ( alphas[i] < C):
                    b = b1
                elif (0 < alphas[j]) and ( alphas[j] < C):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" % \
                      (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
        w = sum(alphas * trainLabelArr * trainDataArr)
    return w, b, alphas

def plotSVM(dataArr, labelArr, w, b):
    xmin = min(dataArr[:, 0])
    xmax = max(dataArr[:, 0])
    x = np.linspace(xmin, xmax, 500)
    y = -w[0] / w[1] * x - b / w[1]
    plt.figure()
    plt.scatter(dataArr[:, 0], dataArr[:, 1], c=(labelArr * 20) + 100)
    plt.plot(x, y)
    curTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    figName = 'simpleSMO_' + curTime + '.png'
    plt.savefig(figName)
    #plt.show()




dataArr, labelArr = createDataSet('testSet.xlsx')
w, b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 3)
plotSVM(dataArr, labelArr, w, b)

