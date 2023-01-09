#!/usr/bin/env python3

import os, math, glob, json, copy, re
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def init():
    os.system("cls")

    
def testAverageCoef():
    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    y = []
    y = list(gatherData["y"].values()) #decoding time of each frame

    X = []  #tool calls for each frame
    for frameNb in gatherData["x"]:
        frameToolCall = list(gatherData["x"][frameNb].values())
        X.append(frameToolCall)

    # ### UVG :
    gatherDataJsonFileUVG = "Gather_Data_UVG.json"
    with open(gatherDataJsonFileUVG) as jf:
        gatherDataUVG = json.load(jf)
    y_UVG = list(gatherDataUVG["y"].values()) #decoding time of each frame
    X_UVG = []  #tool calls for each frame
    for frameNb in gatherDataUVG["x"]:
        frameToolCall = list(gatherDataUVG["x"][frameNb].values())
        X_UVG.append(frameToolCall)

    # The file corresponding to the selected set of coefs
    coefsFile = "coefs/coefs_0_r20.9782639071639995__r2UVG0.8731184983043723_1e-14_3.1622776601682373e-07 1427.json"
    with open(coefsFile) as jf:
        dictCoefs = json.load(jf)
    intercept = dictCoefs["intercept"]
    coefs = dictCoefs["Determined coefs"]

    # CTC
    plotX = []
    plotY = [] 
    print("\n-- my score --")
    myMSE = 0
    for nbFrame in range(len(X)):
        trueTime = y[nbFrame]
        estimatedTime = intercept
        for indexTool in range(len(coefs.keys())):
            estimatedTime += X[nbFrame][indexTool] * list(coefs.values())[indexTool]
        myMSE += (estimatedTime-trueTime)**2 
        plotX.append(trueTime)
        plotY.append(estimatedTime)

    myMSE /= len(X)
    myMSE = math.sqrt(myMSE)
    print("myMSE :",myMSE)

    # UVG
    plotX_uvg = []
    plotY_uvg = [] 
    print("\n-- my score --")
    myMSE = 0
    for nbFrame in range(len(X_UVG)):
        trueTime = y_UVG[nbFrame]
        estimatedTime = intercept
        for indexTool in range(len(coefs.keys())):
            estimatedTime += X_UVG[nbFrame][indexTool] * list(coefs.values())[indexTool]
        myMSE += (estimatedTime-trueTime)**2 
        plotX_uvg.append(trueTime)
        plotY_uvg.append(estimatedTime)

    myMSE /= len(X)
    myMSE = math.sqrt(myMSE)
    print("myMSE UVG :",myMSE)

    plt.scatter(plotX,plotY)
    plt.scatter(plotX_uvg,plotY_uvg)
    plt.scatter(plotX,plotX)
    plt.show()

if __name__ == "__main__": #execute only if ran as a script
    init()
    testAverageCoef()