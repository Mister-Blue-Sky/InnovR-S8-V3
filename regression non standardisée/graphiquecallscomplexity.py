#!/usr/bin/env python3

import os, math, glob, json, copy, re
import numpy as np
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import operator


def init():
    os.system("cls")

def plotCoefsSimple():
    coefsJsonFile = "averages/averageCoefs.json"
    with open(coefsJsonFile) as jf:
        coefs = json.load(jf)

    sortedCoefs = dict(sorted(coefs["Determined coefs"].items(), key=lambda item: item[1]))

    plt.figure(figsize=(320,85))
    plt.bar(sortedCoefs.keys(),sortedCoefs.values())
    plt.xticks(rotation=90,fontsize=10)
    plt.yscale("log")
    plt.savefig("plots/plotCoefs.png")
    plt.show()
    plt.close()

def bigMap():
    # tool complexity
    coefsJsonFile = "averages/averageCoefs.json"
    coefsJsonFile = "coefs/coefs__r20.9794503895968982__r2UVG0.7604056716580654_0.001_1e-06 377_ridge.json"
    with open(coefsJsonFile) as jf:
        toolComplexityDict = json.load(jf)["Determined coefs"]

    # tool use
    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    X = []  #tool calls for each frame
    for frameNb in gatherData["x"]:
        frameToolCall = list(gatherData["x"][frameNb].values())
        X.append(frameToolCall)

    toolUse = copy.deepcopy(toolComplexityDict)
    for tool in toolUse.keys():
        toolUse[tool] = 0
    # get the whole use for each tool
    for frame in range(len(X)):
        for tool in range(len(X[frame])):
            toolUse[list(toolUse.keys())[tool]] += X[frame][tool]
    

    # We split the data in different group
    intraTools = ['intraChromaLumaSubPartitionsHorizontalVertical','intraHVD_HVChromaLuma','intraDCChromaLuma','intraPla_AngLuma','intraHVD_HVLuma','intraHVD_HVChroma','intraChromaLumaSubPartitionsHorizontal','intraChromaLumaSubPartitionsVertical','intraDCLuma','intraPlaChroma','intraDCChroma','intraLumaPDPC','intraChromaMIP','intraDCChroma256','intraDCChroma512','intraDCChroma1024','intraHVDChroma16','intraHVDChroma32','intraHVDChroma64','intraHVDChroma128','intraHVDChroma256','intraHVDChroma512','intraHVDChroma1024','intraHVChroma16','intraHVChroma32','intraHVChroma64','intraHVChroma128','intraHVChroma256','intraHVChroma512','intraHVChroma1024','intraCrossComp16','intraCrossComp32','intraCrossComp64','intraCrossComp128','intraCrossComp256','intraCrossComp512','intraCrossComp1024','intraLumaPDPC16','intraLumaPDPC32','intraLumaPDPC64','intraLumaPDPC128','intraLumaPDPC256','intraLumaPDPC512','intraLumaPDPC1024','intraLumaPDPC2048','intraLumaPDPC4096','intraLumaMIP16','intraLumaMIP32','intraLumaMIP64','intraLumaMIP128','intraLumaMIP256','intraLumaMIP512','intraLumaMIP1024','intraLumaMIP2048','intraLumaMIP4096','intraChromaMIP16','intraChromaMIP32','intraChromaMIP64','intraChromaMIP128','intraChromaMIP256','intraChromaMIP512','intraChromaMIP1024','intraLumaSubPartitionsHorizontal32','intraLumaSubPartitionsHorizontal64','intraLumaSubPartitionsHorizontal128','intraLumaSubPartitionsHorizontal256','intraLumaSubPartitionsHorizontal512','intraLumaSubPartitionsHorizontal1024','intraLumaSubPartitionsHorizontal2048','intraLumaSubPartitionsHorizontal4096','intraChromaSubPartitionsHorizontal16','intraChromaSubPartitionsHorizontal32','intraChromaSubPartitionsHorizontal64','intraChromaSubPartitionsHorizontal128','intraChromaSubPartitionsHorizontal256','intraChromaSubPartitionsHorizontal512','intraChromaSubPartitionsHorizontal1024','intraLumaSubPartitionsVertical32','intraLumaSubPartitionsVertical64','intraLumaSubPartitionsVertical128','intraLumaSubPartitionsVertical256','intraLumaSubPartitionsVertical512','intraLumaSubPartitionsVertical1024','intraLumaSubPartitionsVertical2048','intraLumaSubPartitionsVertical4096','intraChromaSubPartitionsVertical16','intraChromaSubPartitionsVertical32','intraChromaSubPartitionsVertical64','intraChromaSubPartitionsVertical128','intraChromaSubPartitionsVertical256','intraChromaSubPartitionsVertical512','intraChromaSubPartitionsVertical1024','intraPlaLuma16','intraPlaLuma32','intraPlaLuma64','intraPlaLuma128','intraPlaLuma256','intraPlaLuma512','intraPlaLuma1024','intraPlaLuma2048','intraPlaLuma4096','intraDCLuma16','intraDCLuma32','intraDCLuma64','intraDCLuma128','intraDCLuma256','intraDCLuma512','intraDCLuma1024','intraDCLuma2048','intraDCLuma4096','intraHVDLuma16','intraHVDLuma32','intraHVDLuma64','intraHVDLuma128','intraHVDLuma256','intraHVDLuma512','intraHVDLuma1024','intraHVDLuma2048','intraHVDLuma4096','intraHVLuma16','intraHVLuma32','intraHVLuma64','intraHVLuma128','intraHVLuma256','intraHVLuma512','intraHVLuma1024','intraHVLuma2048','intraHVLuma4096','intraAngLuma16','intraAngLuma32','intraAngLuma64','intraAngLuma128','intraAngLuma256','intraAngLuma512','intraAngLuma1024','intraAngLuma2048','intraAngLuma4096','intraPlaChroma16','intraPlaChroma32','intraPlaChroma64','intraPlaChroma128','intraPlaChroma256','intraPlaChroma512','intraPlaChroma1024','intraDCChroma16','intraDCChroma32','intraDCChroma64','intraDCChroma128']
    interTools = ['interChromaLumaMerge_geoChromaLuma','interChromaLumaSkip','fracPelHor_affinefracPelHor_Ver_Both_uni_bi_copyCUPel','interChromaLumaAffine_Skip','inter_transformChromaLumaSkip','geoChromaLuma','interChromaLumaInter','interChromaLumaMerge','','interChromaLumaAffineMerge','interChromaLumaAffineInter','interLumaBlocks32','interLumaBlocks64','interLumaBlocks128','interLumaBlocks256','interLumaBlocks512','interLumaBlocks1024','interLumaBlocks2048','interLumaBlocks4096','interLumaBlocks8192','interLumaBlocks16384','interChromaBlocks8','interChromaBlocks16','interChromaBlocks32','interChromaBlocks64','interChromaBlocks128','interChromaBlocks256','interChromaBlocks512','interChromaBlocks1024','interChromaBlocks2048','interChromaBlocks4096','interLumaInter32','interLumaInter64','interLumaInter128','interLumaInter256','interLumaInter512','interLumaInter1024','interLumaInter2048','interLumaInter4096','uniPredPel','biPredPel','fracPelHor','fracPelVer','fracPelBoth','copyCUPel','affineFracPelHor','affineFracPelVer','affineFracPelBoth','affineCopyCUPel','interLumaInter8192','interLumaInter16384','interChromaInter8','interChromaInter16','interChromaInter32','interChromaInter64','interChromaInter128','interChromaInter256','interChromaInter512','interChromaInter1024','interChromaInter2048','interChromaInter4096','interLumaMerge32','interLumaMerge64','interLumaMerge128','interLumaMerge256','interLumaMerge512','interLumaMerge1024','interLumaMerge2048','interLumaMerge4096','interLumaMerge8192','interLumaMerge16384','interChromaMerge8','interChromaMerge16','interChromaMerge32','interChromaMerge64','interChromaMerge128','interChromaMerge256','interChromaMerge512','interChromaMerge1024','interChromaMerge2048','interChromaMerge4096','interLumaSkip32','interLumaSkip64','interLumaSkip128','interLumaSkip256','interLumaSkip512','interLumaSkip1024','interLumaSkip2048','interLumaSkip4096','interLumaSkip8192','interLumaSkip16384','interChromaSkip8','interChromaSkip16','interChromaSkip32','interChromaSkip64','interChromaSkip128','interChromaSkip256','interChromaSkip512','interChromaSkip1024','interChromaSkip2048','interChromaSkip4096','interLumaAffine64','interLumaAffine128','interLumaAffine256','interLumaAffine512','interLumaAffine1024','interLumaAffine2048','interLumaAffine4096','interLumaAffine8192','interLumaAffine16384','interChromaAffine16','interChromaAffine32','interChromaAffine64','interChromaAffine128','interChromaAffine256','interChromaAffine512','interChromaAffine1024','interChromaAffine2048','interChromaAffine4096','interLumaAffineInter256','interLumaAffineInter512','interLumaAffineInter1024','interLumaAffineInter2048','interLumaAffineInter4096','interLumaAffineInter8192','interLumaAffineInter16384','interChromaAffineInter64','interChromaAffineInter128','interChromaAffineInter256','interChromaAffineInter512','interChromaAffineInter1024','interChromaAffineInter2048','interChromaAffineInter4096','interLumaAffineMerge64','interLumaAffineMerge128','interLumaAffineMerge256','interLumaAffineMerge512','interLumaAffineMerge1024','interLumaAffineMerge2048','interLumaAffineMerge4096','interLumaAffineMerge8192','interLumaAffineMerge16384','interChromaAffineMerge16','interChromaAffineMerge32','interChromaAffineMerge64','interChromaAffineMerge128','interChromaAffineMerge256','interChromaAffineMerge512','interChromaAffineMerge1024','interChromaAffineMerge2048','interChromaAffineMerge4096','interLumaAffineSkip64','interLumaAffineSkip128','interLumaAffineSkip256','interLumaAffineSkip512','interLumaAffineSkip1024','interLumaAffineSkip2048','interLumaAffineSkip4096','interLumaAffineSkip8192','interLumaAffineSkip16384','interChromaAffineSkip16','interChromaAffineSkip32','interChromaAffineSkip64','interChromaAffineSkip128','interChromaAffineSkip256','interChromaAffineSkip512','interChromaAffineSkip1024','interChromaAffineSkip2048','interChromaAffineSkip4096','geoLuma64','geoLuma128','geoLuma256','geoLuma512','geoLuma1024','geoLuma2048','geoLuma4096','geoChroma16','geoChroma32','geoChroma64','geoChroma128','geoChroma256','geoChroma512','geoChroma1024','dmvrBlocks128','dmvrBlocks256','dmvrBlocks512','dmvrBlocks1024','dmvrBlocks2048','dmvrBlocks4096','dmvrBlocks8192','dmvrBlocks16384','bdofBlocks128','bdofBlocks256','bdofBlocks512','bdofBlocks1024','bdofBlocks2048','bdofBlocks4096','bdofBlocks8192','bdof_dmvrBlocks','bdofBlocks16384']
    transformTools = ['transformChromaLumaSkip','transformLFNST_intraLumaMIP','transformChromaLuma','transformLuma16','transformLuma32','transformLuma64','transformLuma128','transformLuma256','transformLuma512','transformLuma1024','transformLuma2048','transformLuma4096','transformChroma4','transformChroma8','transformChroma16','transformChroma32','transformChroma64','transformChroma128','transformChroma256','transformChroma512','transformChroma1024','transformLumaSkip32','transformLumaSkip64','transformLumaSkip128','transformLumaSkip256','transformLumaSkip512','transformLumaSkip1024','transformLumaSkip2048','transformLumaSkip4096','transformChromaSkip8','transformChromaSkip16','transformChromaSkip32','transformChromaSkip64','transformChromaSkip128','transformChromaSkip256','transformChromaSkip512','transformChromaSkip1024','transformLFNST4','transformLFNST8']
    inLoopTools = ['BS0','BS','BS1','BS2','saoLumaBO','saoLumaEO','saoChromaBO','saoChromaEO','alfLumaType7','alfLumaType','alfChromaType','alfChromaType5','ccalf']
    
    dataIntra = {"tool":[],"use":[],"complexity":[],"label":"Intra tools","color":"blue"}
    dataInter = {"tool":[],"use":[],"complexity":[],"label":"Inter tools","color":"black"}
    dataTransform = {"tool":[],"use":[],"complexity":[],"label":"Transform tools","color":"yellow"}
    dataInLoop = {"tool":[],"use":[],"complexity":[],"label":"In-loop tools","color":"red"}

    for tool in toolUse.keys():
        if tool in intraTools:
            dataIntra["tool"].append(tool)
            dataIntra["use"].append(toolUse[tool])
            dataIntra["complexity"].append(toolComplexityDict[tool])
        elif tool in interTools:
            dataInter["tool"].append(tool)
            dataInter["use"].append(toolUse[tool])
            dataInter["complexity"].append(toolComplexityDict[tool])
        elif tool in transformTools:
            dataTransform["tool"].append(tool)
            dataTransform["use"].append(toolUse[tool])
            dataTransform["complexity"].append(toolComplexityDict[tool])
        elif tool in inLoopTools:
            dataInLoop["tool"].append(tool)
            dataInLoop["use"].append(toolUse[tool])
            dataInLoop["complexity"].append(toolComplexityDict[tool])
        else :
            print("Not found :",tool) #impossible

    # We make the scatter plot
    pointsize = 42
    plt.scatter(dataIntra["use"],dataIntra["complexity"],label=dataIntra["label"], s=pointsize, alpha=0.5)
    plt.scatter(dataInter["use"],dataInter["complexity"],label=dataInter["label"], s=pointsize, alpha=0.5)
    plt.scatter(dataTransform["use"],dataTransform["complexity"],label=dataTransform["label"], s=pointsize, alpha=0.5)
    plt.scatter(dataInLoop["use"],dataInLoop["complexity"],label=dataInLoop["label"], s=pointsize, alpha=0.5)
    for i in range(len(toolUse.keys())):
        plt.annotate(list(toolUse.keys())[i], (list(toolUse.values())[i],list(toolComplexityDict.values())[i]))
    plt.title("Complexity depending on the number of calls")
    plt.xlabel("Number of call")
    plt.ylabel("Average complexity per call")
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(fontsize=10, scatterpoints=3)
    
    plt.savefig("plots/bigMap.png",dpi=500)
    plt.show()
    # plt.close()

    # exit()

    ##### pie chart

    labels = "Intra tools","Inter tools","Transform tools","In-loop tools"
    sumComplexity = [0,0,0,0]
    sumCalls = [0,0,0,0]

    for i in range(len(dataIntra['complexity'])):
        sumComplexity[1] += dataIntra["complexity"][i] / dataIntra["use"][i] 
    for i in range(len(dataInter['complexity'])):
        sumComplexity[0] += dataInter["complexity"][i] / dataInter["use"][i] 
    for i in range(len(dataTransform['complexity'])):
        sumComplexity[3] += dataTransform["complexity"][i] / dataTransform["use"][i] 
    for i in range(len(dataInLoop['complexity'])):
        sumComplexity[2] += dataInLoop["complexity"][i] / dataInLoop["use"][i] 

    total = sum(sumComplexity)
    for i in range(len(sumComplexity)):
        sumComplexity[i] = 100 * sumComplexity[i] / total

    print(sumComplexity)
    
    explode = (0.0, 0.0, 0.0, 0.0)
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(tuple(sumComplexity), explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90,pctdistance=0.88, colors=colors)
    # centre_circle = plt.Circle((0,0),0.70,fc='white')
    # fig = plt.gcf()
    # fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    # plt.setp(autotexts, size=8, weight="bold")
    # plt.show()
    plt.savefig("plots/camembert.png")




if __name__ == "__main__": #execute only if ran as a script
    init()
    plotCoefsSimple()
    bigMap()
