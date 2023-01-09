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


def init():
    os.system("cls")


def transformShrinkTool():
    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    newGatherData = {}
    newGatherData["nbBitstream"] = gatherData["nbBitstream"]
    newGatherData["nbFrame"] = gatherData["nbFrame"]
    newGatherData["nbTools"] = 0
    newGatherData["y"] = copy.deepcopy(gatherData["y"])
    newGatherData["x"] = {}

    xEmpty = {}
    for oldTool in gatherData["x"]["0"].keys():
        newToolName = re.findall("[\D]*", oldTool)[0]
        xEmpty[newToolName] = 0

    print(xEmpty)

    for frameId in gatherData["x"].keys():
        newGatherData["x"][frameId] = copy.deepcopy(xEmpty)
        for oldTool in gatherData["x"][frameId].keys():
            newToolName = re.findall("[\D]*", oldTool)[0]
            newGatherData["x"][frameId][newToolName] += gatherData["x"][frameId][oldTool]
   

    newGatherData["nbTools"] = len(newGatherData["x"]["0"].keys())
    print( newGatherData["nbTools"])

    # exit()
    # we dump the result data in a json file
    outFile = "Gather_Data_UVG_new.json"
    with open(outFile, "w") as outfile: 
        json.dump(newGatherData, outfile, indent = 4)

def treatment():

    gatherDataJsonFile = "Gather_Data_CTC.json"
    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    newGatherData = copy.deepcopy(gatherData)

    for frame in gatherData["x"].keys():
        newGatherData["x"][frame].pop("interChromaBlocks")
        newGatherData["x"][frame].pop("interLumaBlocks")

        # newGatherData["x"][frame].pop("BSPel")

        newGatherData["x"][frame].pop("intraCrossComp")

        newGatherData["x"][frame]["interChromaLumaAffineSkip"] = newGatherData["x"][frame]["interChromaAffineSkip"] + newGatherData["x"][frame]["interLumaAffineSkip"]
        newGatherData["x"][frame].pop("interChromaAffineSkip")
        newGatherData["x"][frame].pop("interLumaAffineSkip")
        
        newGatherData["x"][frame]["interChromaLumaInter"] = newGatherData["x"][frame]["interChromaInter"] + newGatherData["x"][frame]["interLumaInter"]
        newGatherData["x"][frame].pop("interChromaInter")
        newGatherData["x"][frame].pop("interLumaInter")
        
        newGatherData["x"][frame]["interChromaLumaMerge"] = newGatherData["x"][frame]["interChromaMerge"] + newGatherData["x"][frame]["interLumaMerge"]
        newGatherData["x"][frame].pop("interChromaMerge")
        newGatherData["x"][frame].pop("interLumaMerge")
        
        newGatherData["x"][frame]["interChromaLumaSkip"] = newGatherData["x"][frame]["interChromaSkip"] + newGatherData["x"][frame]["interLumaSkip"]
        newGatherData["x"][frame].pop("interChromaSkip")
        newGatherData["x"][frame].pop("interLumaSkip")
        
        newGatherData["x"][frame]["transformChromaLuma"] = newGatherData["x"][frame]["transformChroma"] + newGatherData["x"][frame]["transformLuma"]
        newGatherData["x"][frame].pop("transformChroma")
        newGatherData["x"][frame].pop("transformLuma")
        
        newGatherData["x"][frame]["interChromaLumaAffine"] = newGatherData["x"][frame]["interChromaAffine"] + newGatherData["x"][frame]["interLumaAffine"]
        newGatherData["x"][frame].pop("interChromaAffine")
        newGatherData["x"][frame].pop("interLumaAffine")
        
        newGatherData["x"][frame]["interChromaLumaAffineInter"] = newGatherData["x"][frame]["interChromaAffineInter"] + newGatherData["x"][frame]["interLumaAffineInter"]
        newGatherData["x"][frame].pop("interChromaAffineInter")
        newGatherData["x"][frame].pop("interLumaAffineInter")
        
        newGatherData["x"][frame]["interChromaLumaAffineMerge"] = newGatherData["x"][frame]["interChromaAffineMerge"] + newGatherData["x"][frame]["interLumaAffineMerge"]
        newGatherData["x"][frame].pop("interChromaAffineMerge")
        newGatherData["x"][frame].pop("interLumaAffineMerge")
        
        newGatherData["x"][frame]["geoChromaLuma"] = newGatherData["x"][frame]["geoChroma"] + newGatherData["x"][frame]["geoLuma"]
        newGatherData["x"][frame].pop("geoChroma")
        newGatherData["x"][frame].pop("geoLuma")        
        
        newGatherData["x"][frame]["bdof_dmvrBlocks"] = newGatherData["x"][frame]["bdofBlocks"] + newGatherData["x"][frame]["dmvrBlocks"]
        newGatherData["x"][frame].pop("bdofBlocks")
        newGatherData["x"][frame].pop("dmvrBlocks")        
        
        newGatherData["x"][frame]["intraHVD_HVLuma"] = newGatherData["x"][frame]["intraHVDLuma"] + newGatherData["x"][frame]["intraHVLuma"]
        newGatherData["x"][frame].pop("intraHVDLuma")
        newGatherData["x"][frame].pop("intraHVLuma")        
        
        newGatherData["x"][frame]["intraHVD_HVChroma"] = newGatherData["x"][frame]["intraHVDChroma"] + newGatherData["x"][frame]["intraHVChroma"]
        newGatherData["x"][frame].pop("intraHVDChroma")
        newGatherData["x"][frame].pop("intraHVChroma")
        
        newGatherData["x"][frame]["transformChromaLumaSkip"] = newGatherData["x"][frame]["transformLumaSkip"] + newGatherData["x"][frame]["transformChromaSkip"]
        newGatherData["x"][frame].pop("transformLumaSkip")
        newGatherData["x"][frame].pop("transformChromaSkip")
        
        newGatherData["x"][frame]["transformLFNST_intraLumaMIP"] = newGatherData["x"][frame]["transformLFNST"] + newGatherData["x"][frame]["intraLumaMIP"]
        newGatherData["x"][frame].pop("transformLFNST")
        newGatherData["x"][frame].pop("intraLumaMIP")

        newGatherData["x"][frame]["intraPla_AngLuma"] = newGatherData["x"][frame]["intraPlaLuma"] + newGatherData["x"][frame]["intraAngLuma"]
        newGatherData["x"][frame].pop("intraPlaLuma")
        newGatherData["x"][frame].pop("intraAngLuma")


        newGatherData["x"][frame]["fracPelHor_Ver_Both_uni_bi_copyCUPel"] = newGatherData["x"][frame]["fracPelHor"] + newGatherData["x"][frame]["fracPelVer"] + newGatherData["x"][frame]["fracPelBoth"] + newGatherData["x"][frame]["uniPredPel"] + newGatherData["x"][frame]["biPredPel"] + newGatherData["x"][frame]["copyCUPel"]
        newGatherData["x"][frame].pop("fracPelHor")
        newGatherData["x"][frame].pop("fracPelVer")
        newGatherData["x"][frame].pop("fracPelBoth")
        newGatherData["x"][frame].pop("uniPredPel")
        newGatherData["x"][frame].pop("biPredPel")
        newGatherData["x"][frame].pop("copyCUPel")


        newGatherData["x"][frame]["interChromaLumaAffine_Skip"] = newGatherData["x"][frame]["interChromaLumaAffineSkip"] + newGatherData["x"][frame]["interChromaLumaAffine"]
        newGatherData["x"][frame].pop("interChromaLumaAffineSkip")
        newGatherData["x"][frame].pop("interChromaLumaAffine")

        newGatherData["x"][frame]["affineFracPelHor_Ver_Both_copyCUPel"] = newGatherData["x"][frame]["affineFracPelHor"] + newGatherData["x"][frame]["affineFracPelVer"] + newGatherData["x"][frame]["affineFracPelBoth"] + newGatherData["x"][frame]["affineCopyCUPel"]
        newGatherData["x"][frame].pop("affineFracPelHor")
        newGatherData["x"][frame].pop("affineFracPelVer")
        newGatherData["x"][frame].pop("affineFracPelBoth")
        newGatherData["x"][frame].pop("affineCopyCUPel")

        newGatherData["x"][frame]["intraChromaLumaSubPartitionsHorizontal"] = newGatherData["x"][frame]["intraChromaSubPartitionsHorizontal"] + newGatherData["x"][frame]["intraLumaSubPartitionsHorizontal"]
        newGatherData["x"][frame].pop("intraChromaSubPartitionsHorizontal")
        newGatherData["x"][frame].pop("intraLumaSubPartitionsHorizontal")

        newGatherData["x"][frame]["intraChromaLumaSubPartitionsVertical"] = newGatherData["x"][frame]["intraLumaSubPartitionsVertical"] + newGatherData["x"][frame]["intraChromaSubPartitionsVertical"]
        newGatherData["x"][frame].pop("intraLumaSubPartitionsVertical")
        newGatherData["x"][frame].pop("intraChromaSubPartitionsVertical")

        newGatherData["x"][frame]["fracPelHor_affinefracPelHor_Ver_Both_uni_bi_copyCUPel"] = newGatherData["x"][frame]["fracPelHor_Ver_Both_uni_bi_copyCUPel"] + newGatherData["x"][frame]["affineFracPelHor_Ver_Both_copyCUPel"]
        newGatherData["x"][frame].pop("fracPelHor_Ver_Both_uni_bi_copyCUPel")
        newGatherData["x"][frame].pop("affineFracPelHor_Ver_Both_copyCUPel")

        newGatherData["x"][frame]["intraDCChromaLuma"] = newGatherData["x"][frame]["intraDCLuma"] + newGatherData["x"][frame]["intraDCChroma"]
        newGatherData["x"][frame].pop("intraDCLuma")
        newGatherData["x"][frame].pop("intraDCChroma")

        newGatherData["x"][frame]["intraHVD_HVChromaLuma"] = newGatherData["x"][frame]["intraHVD_HVLuma"] + newGatherData["x"][frame]["intraHVD_HVChroma"]
        newGatherData["x"][frame].pop("intraHVD_HVLuma")
        newGatherData["x"][frame].pop("intraHVD_HVChroma")

        newGatherData["x"][frame]["intraChromaLumaSubPartitionsHorizontalVertical"] = newGatherData["x"][frame]["intraChromaLumaSubPartitionsVertical"] + newGatherData["x"][frame]["intraChromaLumaSubPartitionsHorizontal"]
        newGatherData["x"][frame].pop("intraChromaLumaSubPartitionsHorizontal")
        newGatherData["x"][frame].pop("intraChromaLumaSubPartitionsVertical")

        newGatherData["x"][frame]["interChromaLumaMerge_geoChromaLuma"] = newGatherData["x"][frame]["interChromaLumaMerge"] + newGatherData["x"][frame]["geoChromaLuma"]
        newGatherData["x"][frame].pop("interChromaLumaMerge")
        newGatherData["x"][frame].pop("geoChromaLuma")

    newGatherData["nbTools"] = len(newGatherData["x"]["0"].keys())

    print(newGatherData["nbTools"], gatherData["nbTools"] )
    # we dump the result data in a json file
    outFile = "Gather_Data_CTC.json"
    with open(outFile, "w") as outfile: 
        json.dump(newGatherData, outfile, indent = 4)


# Shrinking of the categories
def shrinkTreatment(dataset):

    gatherDataJsonFile = "Gather_Data_{}.json".format(dataset)

    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    newGatherData = copy.deepcopy(gatherData)

    for frame in gatherData["x"].keys():

        dictTransform = {
            "intraBlocks":["intraPlaLuma","intraDCLuma","intraHVDLuma","intraHVLuma","intraAngLuma","intraPlaChroma","intraDCChroma","intraHVDChroma","intraHVChroma","intraCrossComp"],
            "intraPDPC":["intraLumaPDPC"],
            "intraMIP":["intraLumaMIP","intraChromaMIP"],
            "intraSubPartitionsHorizontal":["intraLumaSubPartitionsHorizontal","intraChromaSubPartitionsHorizontal"],
            "intraSubPartitionsVertical":["intraLumaSubPartitionsVertical","intraChromaSubPartitionsVertical"],
            "interInter":["interLumaInter","interChromaInter"],
            "interMerge":["interLumaMerge","interChromaMerge"],
            "interSkip":["interLumaSkip","interChromaSkip"],
            "interAffine":["interLumaAffine","interChromaAffine"],
            "interAffineInter_Merge_Skip":["interLumaAffineInter","interChromaAffineInter","interLumaAffineMerge","interChromaAffineMerge","interLumaAffineSkip","interChromaAffineSkip"],
            "geo":["geoLuma","geoChroma"],
            # dmvrBlocks, bdofBlocks
            "pels":["uniPredPel","biPredPel","fracPelHor","fracPelVer","fracPelBoth","copyCUPel"],
            "affinePels":["affineFracPelHor","affineFracPelVer","affineFracPelBoth","affineCopyCUPel"],
            "transform":["transformLuma","transformChroma"],
            "transformSkip":["transformLumaSkip","transformChromaSkip"],
            "sao":["saoLumaBO","saoLumaEO","saoChromaBO","saoChromaEO"],
            "alf":["alfLumaType","alfChromaType","ccalf"]}

        for newTool in dictTransform.keys():
            tempSum = 0
            for tool in dictTransform[newTool]:
                tempSum += newGatherData["x"][frame][tool]
                newGatherData["x"][frame].pop(tool)

            newGatherData["x"][frame][newTool] = tempSum
      
    newGatherData["nbTools"] = len(newGatherData["x"]["0"].keys())

    print(newGatherData["nbTools"], gatherData["nbTools"] )
    # we dump the result data in a json file
    outFile = "Gather_Data_{}_shrink.json".format(dataset)
    with open(outFile, "w") as outfile: 
        json.dump(newGatherData, outfile, indent = 4)

def correction(dataset):
    gatherDataJsonFile = "Gather_Data_{}_shrink.json".format(dataset)

    with open(gatherDataJsonFile) as jf:
        gatherData = json.load(jf)

    newGatherData = copy.deepcopy(gatherData)

    for frame in gatherData["x"].keys():
        newGatherData["x"][frame]["pels"] += newGatherData["x"][frame]["affinePels"]
        newGatherData["x"][frame].pop("affinePels")
        newGatherData["x"][frame].pop("interAffine")


        # newGatherData["x"][frame]["interChromaLumaMerge_geoChromaLuma"] = newGatherData["x"][frame]["interChromaLumaMerge"] + newGatherData["x"][frame]["geoChromaLuma"]
        # newGatherData["x"][frame].pop("interChromaLumaMerge")
        # newGatherData["x"][frame].pop("geoChromaLuma")
        
      
    newGatherData["nbTools"] = len(newGatherData["x"]["0"].keys())

    print(newGatherData["nbTools"], gatherData["nbTools"] )
    # we dump the result data in a json file
    outFile = "Gather_Data_{}_shrink_v2.json".format(dataset)
    with open(outFile, "w") as outfile: 
        json.dump(newGatherData, outfile, indent = 4)

if __name__ == "__main__": #execute only if ran as a script
    init()

    transformShrinkTool()

    # shrinkTreatment("CTC")
    # shrinkTreatment("UVG")

    correction("CTC")
    correction("UVG")

    # treatment()

