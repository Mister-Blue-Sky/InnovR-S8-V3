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
import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

def init():
    os.system("cls")


def corrMatrix():

    ########## dataset handling ##########
    print("dataset handling...")
    
    ### CTC :
    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    y = list(gatherData["y"].values()) #decoding time of each frame

    X = []  #tool calls for each frame
    for frameNb in gatherData["x"]:
        frameToolCall = list(gatherData["x"][frameNb].values())
        X.append(frameToolCall)

    # corr matrix
    print("Corr matrix")
    df = pd.DataFrame(X,columns = list(gatherData["x"]["0"].keys()))
    df["time"] = y
    plt.figure(figsize=(220,220))
    cor = df.corr('spearman')
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Blues,norm=PowerNorm(0.95))
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
    plt.savefig("heatmaps/heatmap_CTC_spearman_nonorm.png")
    plt.close()


if __name__ == "__main__": #execute only if ran as a script
    init()
    corrMatrix()



