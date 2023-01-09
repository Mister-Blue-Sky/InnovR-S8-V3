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
from matplotlib.lines import Line2D

def init():
    os.system("cls")

def average():
    newCoefs = {}
    newCoefs["intercept"] = 0
    newCoefs["Determined coefs"] = {}
    for rdmcoefsFile in glob.glob("coefs/*.json"):
        with open(rdmcoefsFile) as jf:
            rdmcoefsDict = json.load(jf)
        for tool in rdmcoefsDict["Determined coefs"].keys():
            newCoefs["Determined coefs"][tool] = 0
        break
    
    nbFile = 0
    for coefsFile in glob.glob("coefs/*.json"):
        with open(coefsFile) as jf:
            coefsDict = json.load(jf)
        
        nbFile += 1

        for tool in newCoefs["Determined coefs"].keys():
            newCoefs["Determined coefs"][tool] += coefsDict["Determined coefs"][tool]
        newCoefs["intercept"] += coefsDict["intercept"]
        
    for tool in newCoefs["Determined coefs"].keys():
        newCoefs["Determined coefs"][tool] /= nbFile
    newCoefs["intercept"] /= nbFile

    outFile = "averages/averageCoefs.json"
    with open(outFile, "w") as outfile: 
        json.dump(newCoefs, outfile, indent = 4)

def plotDistribution():

    data = {}
    for coefsFile in glob.glob("coefs/*.json"):
        with open(coefsFile) as jf:
            coefsDict = json.load(jf)
        if coefsDict["nbZero"] != 0:
            continue
        if "interAffine" not in list(coefsDict["Determined coefs"].keys()):
            continue
        data[coefsFile] = coefsDict["Determined coefs"]

    for fileName in data.keys():
        data[fileName]["pels"] += data[fileName]["affinePels"] 
        data[fileName].pop("affinePels")
        if "interAffine" in list(data[fileName].keys()):
            data[fileName].pop("interAffine")
        

    ### tool order
    dataTemp = {}
    intra = ['intraBlocks',  'intraMIP','intraPDPC', 'intraSubPartitionsHorizontal', 'intraSubPartitionsVertical',]
    inter = [ 'interChromaBlocks', 'interMerge','interLumaBlocks', 'interSkip','interInter','dmvrBlocks', 'geo', 'interAffineInter_Merge_Skip','bdofBlocks', 'pels']
    transform = ['transform','transformLFNST', 'transformSkip',]
    inloop = ['alf','BS','sao']
    toolOrder = inloop + transform + inter + intra 
    print(toolOrder)

    for fileName in data.keys():
        dataTemp[fileName] = {}
        for tool in toolOrder:
            dataTemp[fileName][tool] = data[fileName][tool] 

    data.clear()
    data = copy.deepcopy(dataTemp)
    ###

    means = {}
    ecarttypes = {}
    for tool in data[list(data.keys())[0]].keys():
        means[tool] = 0
        ecarttypes[tool] = 0
    
    for fileName in data.keys():
        for tool in data[fileName].keys():
            means[tool] += data[fileName][tool]
    for tool in means.keys():
        means[tool] /= len(means.keys())

    for fileName in data.keys():
        for tool in data[fileName].keys():
            ecarttypes[tool] += (data[fileName][tool] - means[tool])**2
    for tool in ecarttypes.keys():
        ecarttypes[tool] = math.sqrt(ecarttypes[tool]/len(ecarttypes.keys()))

    errors = {}
    for tool in ecarttypes.keys():
        errors[tool] = 1.96 * ecarttypes[tool] / math.sqrt(len(ecarttypes.keys()))


    print(means.keys())
    custom_lines = [Line2D([0], [0], color="blue", lw=4),
                Line2D([0], [0], color="red", lw=4),
                Line2D([0], [0], color="green", lw=4),
                Line2D([0], [0], color="orange", lw=4)]

    colors = ['orange'] * len(inloop) + ['green'] * len(transform) + ['red'] * len(inter) + ['blue'] * len(intra)
    plt.barh(list(means.keys()), list(means.values()),color=colors,xerr=list(errors.values()))
    plt.legend(custom_lines,['Intra','Inter','Transform','In-loop'])
    plt.xscale('log')
    
    # plt.ticks(rotation=90,fontsize=10)
    plt.gcf().subplots_adjust(left = 0.35)
    plt.show()

    #### map ######################

    # toolComplexityDict = copy.deepcopy(means)

    # tool use
    gatherDataJsonFile = "Gather_Data_CTC_Divide10_smartShrink_presque.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    toolCalls = {}
    for tool in means.keys():
        toolCalls[tool] = 0
    
    for frameNb in gatherData["x"].keys():
        for tool in gatherData["x"][frameNb].keys():
            toolCalls[tool] += gatherData["x"][frameNb][tool]

    print(toolCalls)

    # X = []  #tool calls for each frame
    # for frameNb in gatherData["x"]:
    #     frameToolCall = list(gatherData["x"][frameNb].values())
    #     X.append(frameToolCall)

    # toolUse = copy.deepcopy(toolComplexityDict)
    # for tool in toolUse.keys():
    #     toolUse[tool] = 0
    # # get the whole use for each tool
    # for frame in range(len(X)):
    #     for tool in range(len(X[frame])):
    #         toolUse[list(toolUse.keys())[tool]] += X[frame][tool]
    

    # We split the data in different group
    # intraTools = ['intraChromaLumaSubPartitionsHorizontalVertical','intraHVD_HVChromaLuma','intraDCChromaLuma','intraPla_AngLuma','intraHVD_HVLuma','intraHVD_HVChroma','intraChromaLumaSubPartitionsHorizontal','intraChromaLumaSubPartitionsVertical','intraDCLuma','intraPlaChroma','intraDCChroma','intraLumaPDPC','intraChromaMIP','intraDCChroma256','intraDCChroma512','intraDCChroma1024','intraHVDChroma16','intraHVDChroma32','intraHVDChroma64','intraHVDChroma128','intraHVDChroma256','intraHVDChroma512','intraHVDChroma1024','intraHVChroma16','intraHVChroma32','intraHVChroma64','intraHVChroma128','intraHVChroma256','intraHVChroma512','intraHVChroma1024','intraCrossComp16','intraCrossComp32','intraCrossComp64','intraCrossComp128','intraCrossComp256','intraCrossComp512','intraCrossComp1024','intraLumaPDPC16','intraLumaPDPC32','intraLumaPDPC64','intraLumaPDPC128','intraLumaPDPC256','intraLumaPDPC512','intraLumaPDPC1024','intraLumaPDPC2048','intraLumaPDPC4096','intraLumaMIP16','intraLumaMIP32','intraLumaMIP64','intraLumaMIP128','intraLumaMIP256','intraLumaMIP512','intraLumaMIP1024','intraLumaMIP2048','intraLumaMIP4096','intraChromaMIP16','intraChromaMIP32','intraChromaMIP64','intraChromaMIP128','intraChromaMIP256','intraChromaMIP512','intraChromaMIP1024','intraLumaSubPartitionsHorizontal32','intraLumaSubPartitionsHorizontal64','intraLumaSubPartitionsHorizontal128','intraLumaSubPartitionsHorizontal256','intraLumaSubPartitionsHorizontal512','intraLumaSubPartitionsHorizontal1024','intraLumaSubPartitionsHorizontal2048','intraLumaSubPartitionsHorizontal4096','intraChromaSubPartitionsHorizontal16','intraChromaSubPartitionsHorizontal32','intraChromaSubPartitionsHorizontal64','intraChromaSubPartitionsHorizontal128','intraChromaSubPartitionsHorizontal256','intraChromaSubPartitionsHorizontal512','intraChromaSubPartitionsHorizontal1024','intraLumaSubPartitionsVertical32','intraLumaSubPartitionsVertical64','intraLumaSubPartitionsVertical128','intraLumaSubPartitionsVertical256','intraLumaSubPartitionsVertical512','intraLumaSubPartitionsVertical1024','intraLumaSubPartitionsVertical2048','intraLumaSubPartitionsVertical4096','intraChromaSubPartitionsVertical16','intraChromaSubPartitionsVertical32','intraChromaSubPartitionsVertical64','intraChromaSubPartitionsVertical128','intraChromaSubPartitionsVertical256','intraChromaSubPartitionsVertical512','intraChromaSubPartitionsVertical1024','intraPlaLuma16','intraPlaLuma32','intraPlaLuma64','intraPlaLuma128','intraPlaLuma256','intraPlaLuma512','intraPlaLuma1024','intraPlaLuma2048','intraPlaLuma4096','intraDCLuma16','intraDCLuma32','intraDCLuma64','intraDCLuma128','intraDCLuma256','intraDCLuma512','intraDCLuma1024','intraDCLuma2048','intraDCLuma4096','intraHVDLuma16','intraHVDLuma32','intraHVDLuma64','intraHVDLuma128','intraHVDLuma256','intraHVDLuma512','intraHVDLuma1024','intraHVDLuma2048','intraHVDLuma4096','intraHVLuma16','intraHVLuma32','intraHVLuma64','intraHVLuma128','intraHVLuma256','intraHVLuma512','intraHVLuma1024','intraHVLuma2048','intraHVLuma4096','intraAngLuma16','intraAngLuma32','intraAngLuma64','intraAngLuma128','intraAngLuma256','intraAngLuma512','intraAngLuma1024','intraAngLuma2048','intraAngLuma4096','intraPlaChroma16','intraPlaChroma32','intraPlaChroma64','intraPlaChroma128','intraPlaChroma256','intraPlaChroma512','intraPlaChroma1024','intraDCChroma16','intraDCChroma32','intraDCChroma64','intraDCChroma128']
    # interTools = ['interChromaLumaMerge_geoChromaLuma','interChromaLumaSkip','fracPelHor_affinefracPelHor_Ver_Both_uni_bi_copyCUPel','interChromaLumaAffine_Skip','inter_transformChromaLumaSkip','geoChromaLuma','interChromaLumaInter','interChromaLumaMerge','','interChromaLumaAffineMerge','interChromaLumaAffineInter','interLumaBlocks32','interLumaBlocks64','interLumaBlocks128','interLumaBlocks256','interLumaBlocks512','interLumaBlocks1024','interLumaBlocks2048','interLumaBlocks4096','interLumaBlocks8192','interLumaBlocks16384','interChromaBlocks8','interChromaBlocks16','interChromaBlocks32','interChromaBlocks64','interChromaBlocks128','interChromaBlocks256','interChromaBlocks512','interChromaBlocks1024','interChromaBlocks2048','interChromaBlocks4096','interLumaInter32','interLumaInter64','interLumaInter128','interLumaInter256','interLumaInter512','interLumaInter1024','interLumaInter2048','interLumaInter4096','uniPredPel','biPredPel','fracPelHor','fracPelVer','fracPelBoth','copyCUPel','affineFracPelHor','affineFracPelVer','affineFracPelBoth','affineCopyCUPel','interLumaInter8192','interLumaInter16384','interChromaInter8','interChromaInter16','interChromaInter32','interChromaInter64','interChromaInter128','interChromaInter256','interChromaInter512','interChromaInter1024','interChromaInter2048','interChromaInter4096','interLumaMerge32','interLumaMerge64','interLumaMerge128','interLumaMerge256','interLumaMerge512','interLumaMerge1024','interLumaMerge2048','interLumaMerge4096','interLumaMerge8192','interLumaMerge16384','interChromaMerge8','interChromaMerge16','interChromaMerge32','interChromaMerge64','interChromaMerge128','interChromaMerge256','interChromaMerge512','interChromaMerge1024','interChromaMerge2048','interChromaMerge4096','interLumaSkip32','interLumaSkip64','interLumaSkip128','interLumaSkip256','interLumaSkip512','interLumaSkip1024','interLumaSkip2048','interLumaSkip4096','interLumaSkip8192','interLumaSkip16384','interChromaSkip8','interChromaSkip16','interChromaSkip32','interChromaSkip64','interChromaSkip128','interChromaSkip256','interChromaSkip512','interChromaSkip1024','interChromaSkip2048','interChromaSkip4096','interLumaAffine64','interLumaAffine128','interLumaAffine256','interLumaAffine512','interLumaAffine1024','interLumaAffine2048','interLumaAffine4096','interLumaAffine8192','interLumaAffine16384','interChromaAffine16','interChromaAffine32','interChromaAffine64','interChromaAffine128','interChromaAffine256','interChromaAffine512','interChromaAffine1024','interChromaAffine2048','interChromaAffine4096','interLumaAffineInter256','interLumaAffineInter512','interLumaAffineInter1024','interLumaAffineInter2048','interLumaAffineInter4096','interLumaAffineInter8192','interLumaAffineInter16384','interChromaAffineInter64','interChromaAffineInter128','interChromaAffineInter256','interChromaAffineInter512','interChromaAffineInter1024','interChromaAffineInter2048','interChromaAffineInter4096','interLumaAffineMerge64','interLumaAffineMerge128','interLumaAffineMerge256','interLumaAffineMerge512','interLumaAffineMerge1024','interLumaAffineMerge2048','interLumaAffineMerge4096','interLumaAffineMerge8192','interLumaAffineMerge16384','interChromaAffineMerge16','interChromaAffineMerge32','interChromaAffineMerge64','interChromaAffineMerge128','interChromaAffineMerge256','interChromaAffineMerge512','interChromaAffineMerge1024','interChromaAffineMerge2048','interChromaAffineMerge4096','interLumaAffineSkip64','interLumaAffineSkip128','interLumaAffineSkip256','interLumaAffineSkip512','interLumaAffineSkip1024','interLumaAffineSkip2048','interLumaAffineSkip4096','interLumaAffineSkip8192','interLumaAffineSkip16384','interChromaAffineSkip16','interChromaAffineSkip32','interChromaAffineSkip64','interChromaAffineSkip128','interChromaAffineSkip256','interChromaAffineSkip512','interChromaAffineSkip1024','interChromaAffineSkip2048','interChromaAffineSkip4096','geoLuma64','geoLuma128','geoLuma256','geoLuma512','geoLuma1024','geoLuma2048','geoLuma4096','geoChroma16','geoChroma32','geoChroma64','geoChroma128','geoChroma256','geoChroma512','geoChroma1024','dmvrBlocks128','dmvrBlocks256','dmvrBlocks512','dmvrBlocks1024','dmvrBlocks2048','dmvrBlocks4096','dmvrBlocks8192','dmvrBlocks16384','bdofBlocks128','bdofBlocks256','bdofBlocks512','bdofBlocks1024','bdofBlocks2048','bdofBlocks4096','bdofBlocks8192','bdof_dmvrBlocks','bdofBlocks16384']
    # transformTools = ['transformChromaLumaSkip','transformLFNST_intraLumaMIP','transformChromaLuma','transformLuma16','transformLuma32','transformLuma64','transformLuma128','transformLuma256','transformLuma512','transformLuma1024','transformLuma2048','transformLuma4096','transformChroma4','transformChroma8','transformChroma16','transformChroma32','transformChroma64','transformChroma128','transformChroma256','transformChroma512','transformChroma1024','transformLumaSkip32','transformLumaSkip64','transformLumaSkip128','transformLumaSkip256','transformLumaSkip512','transformLumaSkip1024','transformLumaSkip2048','transformLumaSkip4096','transformChromaSkip8','transformChromaSkip16','transformChromaSkip32','transformChromaSkip64','transformChromaSkip128','transformChromaSkip256','transformChromaSkip512','transformChromaSkip1024','transformLFNST4','transformLFNST8']
    # inLoopTools = ['BS0','BS','BS1','BS2','saoLumaBO','saoLumaEO','saoChromaBO','saoChromaEO','alfLumaType7','alfLumaType','alfChromaType','alfChromaType5','ccalf']
    
    intraTools = copy.deepcopy(intra)
    interTools = copy.deepcopy(inter)
    transformTools = copy.deepcopy(transform)
    inLoopTools = copy.deepcopy(inloop)

    dataIntra = {"tool":[],"use":[],"complexity":[],"label":"Intra tools","color":"blue"}
    dataInter = {"tool":[],"use":[],"complexity":[],"label":"Inter tools","color":"red"}
    dataTransform = {"tool":[],"use":[],"complexity":[],"label":"Transform tools","color":"green"}
    dataInLoop = {"tool":[],"use":[],"complexity":[],"label":"In-loop tools","color":"orange"}

    for tool in toolCalls.keys():
        if tool in intraTools:
            dataIntra["tool"].append(tool)
            dataIntra["use"].append(toolCalls[tool])
            dataIntra["complexity"].append(means[tool])
        elif tool in interTools:
            dataInter["tool"].append(tool)
            dataInter["use"].append(toolCalls[tool])
            dataInter["complexity"].append(means[tool])
        elif tool in transformTools:
            dataTransform["tool"].append(tool)
            dataTransform["use"].append(toolCalls[tool])
            dataTransform["complexity"].append(means[tool])
        elif tool in inLoopTools:
            dataInLoop["tool"].append(tool)
            dataInLoop["use"].append(toolCalls[tool])
            dataInLoop["complexity"].append(means[tool])
        else :
            print("Not found :",tool) #impossible


    # We make the scatter plot
    pointsize = 42
    plt.scatter(dataIntra["use"],dataIntra["complexity"],label=dataIntra["label"], s=pointsize, alpha=1,c="blue")
    plt.scatter(dataInter["use"],dataInter["complexity"],label=dataInter["label"], s=pointsize, alpha=1,c="red")
    plt.scatter(dataTransform["use"],dataTransform["complexity"],label=dataTransform["label"], s=pointsize, alpha=1,c="green")
    plt.scatter(dataInLoop["use"],dataInLoop["complexity"],label=dataInLoop["label"], s=pointsize, alpha=1,c="orange")
    for i in range(len(toolCalls.keys())):
        plt.annotate(list(toolCalls.keys())[i], (list(toolCalls.values())[i],list(means.values())[i]))
    plt.title("Complexity depending on the number of calls")
    plt.xlabel("Number of call")
    plt.ylabel("Average complexity per call")
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.legend(fontsize=10, scatterpoints=3)
    
    # plt.savefig("plots/bigMap.png",dpi=500)
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
    colors = ['blue','red','green','orange']

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(tuple(sumComplexity), explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=False, startangle=90,pctdistance=0.82, colors=colors)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.tight_layout()
    # plt.setp(autotexts, size=8, weight="bold")
    plt.show()
    # plt.savefig("plots/camembert.png")






if __name__ == "__main__": #execute only if ran as a script
    init()
    average()
    plotDistribution()
