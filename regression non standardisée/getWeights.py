#!/usr/bin/env python3
import os, math, glob, json
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

def init():
    os.system("cls")


def getWeights(alpha, tol, index, maxIter):
    
    ########## dataset handling ##########
    print("dataset handling...")
    print("alpha :",alpha,"| tol :",tol)
    
    ### CTC :
    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    y = list(gatherData["y"].values()) #decoding time of each frame

    X = []  #tool calls for each frame
    for frameNb in gatherData["x"]:
        frameToolCall = list(gatherData["x"][frameNb].values())
        X.append(frameToolCall)


    ### UVG :
    gatherDataJsonFileUVG = "Gather_Data_UVG.json"
    with open(gatherDataJsonFileUVG) as jf:
        gatherDataUVG = json.load(jf)

    y_UVG = list(gatherDataUVG["y"].values()) #decoding time of each frame

    X_UVG = []  #tool calls for each frame
    for frameNb in gatherDataUVG["x"]:
        frameToolCall = list(gatherDataUVG["x"][frameNb].values())
        X_UVG.append(frameToolCall)

    ###

    for d in y_UVG:
        y.append(d)
    for d in X_UVG:
        X.append(d)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    ######### linear regression ##########
    print("linear regression ...")

    model = linear_model.Ridge(alpha=alpha,max_iter=maxIter,tol=tol, positive=True, copy_X=True)
    param = str(alpha) + "_" + str(tol) + " " + str(index)"
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # scores :
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test,y_pred,squared=False)
    print("model R2 score", r2)
    print("model root mean squared error score", mse)


    # weights :
    weights = model.coef_
    intercept = model.intercept_

    ### Ultra Video Group dataset score ##
    y_pred_UVG = model.predict(X_UVG)
    
    # scores :
    r2_UVG = r2_score(y_UVG, y_pred_UVG)
    mse_UVG = mean_squared_error(y_UVG,y_pred_UVG,squared=False)
    print("UVG model R2 score", r2_UVG)
    print("UVG model root mean squared error score", mse_UVG)


    tmpNbZero = 0
    for value in list(weights):
        if(value == 0.0):
            tmpNbZero += 1

    ########### export results ###########

    result = {}
    result["gatherData file"] = gatherDataJsonFile
    result["testing group"] = "20% of the original dataset"
    result["alpha"] = alpha
    result["tol"] = tol
    result["maxIter"] = maxIter
    result["mse"] = mse
    result["r2"] = r2
    result["intercept"] = intercept
    result["mse_uvg"] = mse_UVG
    result["r2_uvg"] = r2_UVG
    result["maxIter"] = maxIter
    result["nbZero"] = tmpNbZero
    result["Determined coefs"] = dict(zip(list(gatherData['x']['0'].keys()),list(weights)))

    # we dump the result data in a json file
    with open("coefs/coefs_" + str(tmpNbZero) + "_r2" + str(r2) + "__r2UVG" + str(r2_UVG) + "_" + param + ".json", "w") as outfile: 
        json.dump(result, outfile, indent = 4)


if __name__ == "__main__": #execute only if ran as a script
    init()

    # We test different combinations of hyperparameters
    maxIter = 10**8
    index = 0
    for a in np.arange(-20,-12,0.5).astype(float): #0.3
        alpha = 10**a
        # alpha = 10**-16
        for t in np.arange(-8,-5,0.1).astype(float): #1
            tol = 10**t
            for _ in range(10):
                getWeights(alpha,tol,index,maxIter)
                index += 1



