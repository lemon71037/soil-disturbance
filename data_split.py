import numpy as np

def calSingleCorrelation(a, b):
    """计算a b两个信号的余弦相关性
    """
    numerator = 0
    dedominator = 0
    tempA = 0
    tempB = 0
    for i, j in zip(a, b):
        numerator += i*j
        tempA += i*i
        tempB += j*j
    dedominator = np.sqrt(tempA*tempB)
    return np.abs(numerator)/dedominator

def calCorrelation(a, b, windowSize, stepSize):
    """按滑动窗口，计算a b信号的余弦相似性
    """
    corr = []
    for i in range(0, len(a), stepSize):
        corr.append(calSingleCorrelation(a[i: i+windowSize], b[i: i+windowSize]))
    return corr

def activitySplit(dataXYZ, windowSize, stepSize, corrThresholdLength):
    """以余弦相关性来切割事件
    """
    joinWindowSize = windowSize  # 两个窗口拼接成一个时间的阈值
    # folderPath = r"D:\研一\土壤扰动\syf\dig"
    # csvPath = folderPath + "\\" + csvName
    # dataXYZ = pd.read_csv(csvPath, header= 0)
    corrXYs = calCorrelation(dataXYZ.iloc[:, 0], dataXYZ.iloc[:, 1], windowSize, stepSize) 
    activityPosList = []
    isFirst = True
    
    corrThresholdList = corrXYs[: corrThresholdLength]
    corrThresholdListCopy = corrThresholdList.copy()
    corrThresholdListCopy.sort()
    percent25 = corrThresholdListCopy[int(0.25*corrThresholdLength)]
    percent75 = corrThresholdListCopy[int(0.75*corrThresholdLength)]
    
    coef = 3
    corrThreshold = percent25 - coef*(np.abs(percent75 - percent25))
    #corrThreshold = np.abs(percent75 + coef*(np.abs(percent75 - percent25)))
    
    for i, corrXY in enumerate(corrXYs):
        if np.abs(corrXY) <= corrThreshold:
            if isFirst:
                start = i * stepSize
                end = start + windowSize
                isFirst = False
            else:
                if i*stepSize - end <= joinWindowSize:
                    end = i*stepSize + windowSize
                else:
                    activityPosList.append([start, end])
                    start = i * stepSize
                    end = start + windowSize
        corrThresholdList.pop(0)
        corrThresholdList.append(corrXY)
        corrThresholdListCopy = corrThresholdList.copy()
        corrThresholdListCopy.sort()
        percent25 = corrThresholdListCopy[int(0.25*corrThresholdLength)]
        percent75 = corrThresholdListCopy[int(0.75*corrThresholdLength)]
        corrThreshold = percent25 - coef*(np.abs(percent75 - percent25))
        # corrThreshold = np.abs(percent75 + coef*(np.abs(percent75 - percent25)))

    # print(csvName, len(activityPosList))
    return activityPosList
