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
    gatherDataJsonFile = "Gather_Data_CTC_Full_Only_used_tool.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    y = []
    y = list(gatherData["y"].values()) #decoding time of each frame

    X = []  #tool calls for each frame
    for frameNb in gatherData["x"]:
        frameToolCall = list(gatherData["x"][frameNb].values())
        X.append(frameToolCall)

    scaler = StandardScaler(copy=True,with_mean=True,with_std=True)
    scaler.fit(X)
    X = scaler.transform(X)

    # ### UVG :
    gatherDataJsonFileUVG = "Gather_Data_UVG_Full_Only_used_tool.json"
    with open(gatherDataJsonFileUVG) as jf:
        gatherDataUVG = json.load(jf)
    y_UVG = list(gatherDataUVG["y"].values()) #decoding time of each frame
    X_UVG = []  #tool calls for each frame
    for frameNb in gatherDataUVG["x"]:
        frameToolCall = list(gatherDataUVG["x"][frameNb].values())
        X_UVG.append(frameToolCall)
    X_UVG = scaler.transform(X_UVG)

    # The file corresponding to the selected set of coefs
    coefsFile = "finalCoefs\r20.992134439342894__r2UVG0.9027066812793761_12589.254117941637_100000000  18_ridge.json"
    with open(coefsFile) as jf:
        dictCoefs = json.load(jf)
    intercept = dictCoefs["intercept"]
    coefs = dictCoefs["Determined coefs"]

    # CTC
    plotX = []
    plotY = [] 
    print("\n-- my score --")
    myMSE = 0
    estimationError = 0
    myP = []
    myPP = []
    for nbFrame in range(len(X)):
        trueTime = y[nbFrame]
        estimatedTime = intercept
        for indexTool in range(len(coefs.keys())):
            estimatedTime += X[nbFrame][indexTool] * list(coefs.values())[indexTool]

        myMSE += (estimatedTime-trueTime)**2 
        estimationError += math.fabs((trueTime-estimatedTime)/trueTime)
        plotX.append(trueTime)
        plotY.append(estimatedTime)
        myP.append(((trueTime-estimatedTime)/trueTime)**2)
        myPP.append((estimatedTime-trueTime)**2 )

    myMSE /= len(X)
    estimationError /= len(X)
    myMSE = math.sqrt(myMSE)
    print("myMSE :",myMSE)
    print("myMSEp :",estimationError*100,"%")

    # UVG
    plotX_uvg = []
    plotY_uvg = [] 
    print("\n-- my score --")
    myMSE = 0
    estimationError = 0
    for nbFrame in range(len(X_UVG)):
        trueTime = y_UVG[nbFrame]
        estimatedTime = intercept
        for indexTool in range(len(coefs.keys())):
            estimatedTime += X_UVG[nbFrame][indexTool] * list(coefs.values())[indexTool]
        myMSE += (estimatedTime-trueTime)**2 
        estimationError += math.fabs((trueTime-estimatedTime)/trueTime)
        plotX_uvg.append(trueTime)
        plotY_uvg.append(estimatedTime)

    myMSE /= len(X_UVG)
    estimationError /= len(X_UVG)
    myMSE = math.sqrt(myMSE)
    print("myMSE UVG :",myMSE)
    print("myMSEp UVG :",estimationError*100,"%")

    fig, ax = plt.subplots()
    ax.scatter(plotX,plotY,label="CTC")
    ax.scatter(plotX_uvg,plotY_uvg,label="UVG")
    ax.plot([0, 12/5], [0, 12/5], color = 'red', linestyle = 'solid',label="$\^t_{dec} = t_{dec}$")
    plt.xlabel('$t_{dec} in seconds$')
    plt.ylabel('$\^t_{dec} in seconds$')
    plt.legend()
    # plt.savefig("standardiseeregression.png",dpi=500)
    plt.show()

if __name__ == "__main__": #execute only if ran as a script
    init()
    testAverageCoef()