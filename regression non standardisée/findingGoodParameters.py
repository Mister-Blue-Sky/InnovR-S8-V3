import glob, json,os,copy
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

os.system("cls")

Aalphas = []
Atols = []
Ar2s = []
Ar2sUVG = []
Amse = []
AmseUVG = []
AnbZero = []

for jsonFile in glob.glob("coefs/*.json"):
    with open(jsonFile) as jf:
        dataDict = json.load(jf)
        Aalphas.append(dataDict["alpha"])
        Atols.append(dataDict["tol"])
        Ar2s.append(dataDict["r2"])
        Ar2sUVG.append(dataDict["r2_uvg"])
        Amse.append(dataDict["mse"])
        AmseUVG.append(dataDict["mse_uvg"])
        tmpNbZero = 0
        for value in list(dataDict["Determined coefs"].values()):
            if(value == 0.0):
                tmpNbZero += 1
        AnbZero.append(tmpNbZero)

setAlpha = list(set(Aalphas))
setTols = list(set(Atols))
data = {}
for a in setAlpha:
    data[a] = {}
    for t in setTols:
        data[a][t] = {}
        data[a][t]["nb"] = 0
        data[a][t]["r2"] = 0
        data[a][t]["r2uvg"] = 0
        data[a][t]["mse"] = 0
        data[a][t]["mseuvg"] = 0
        data[a][t]["nbzero"] = 0

for i in range(len(Aalphas)):
    data[Aalphas[i]][Atols[i]]["nb"] += 1
    data[Aalphas[i]][Atols[i]]["r2"] += copy.deepcopy(Ar2s[i])
    data[Aalphas[i]][Atols[i]]["r2uvg"] += copy.deepcopy(Ar2sUVG[i])
    data[Aalphas[i]][Atols[i]]["mse"] += copy.deepcopy(Amse[i])
    data[Aalphas[i]][Atols[i]]["mseuvg"] += copy.deepcopy(AmseUVG[i])
    data[Aalphas[i]][Atols[i]]["nbzero"] += copy.deepcopy(AnbZero[i])

alphas = []
tols = []
r2s = []
r2sUVG = []
mse = []
mseUVG = []
nbZero = []

for a in data.keys():
    for t in data[a].keys():
        if data[a][t]["nb"] == 0:
            print("Rare")
            continue
        data[a][t]["r2"] /= data[a][t]["nb"]
        data[a][t]["r2uvg"] /= data[a][t]["nb"]
        data[a][t]["mse"] /= data[a][t]["nb"]
        data[a][t]["mseuvg"] /= data[a][t]["nb"]
        data[a][t]["nbzero"] /= data[a][t]["nb"]

        alphas.append(a)
        tols.append(t)
        r2s.append(data[a][t]["r2"])
        r2sUVG.append(data[a][t]["r2uvg"])
        mse.append(data[a][t]["mse"])
        mseUVG.append(data[a][t]["mseuvg"])
        nbZero.append(data[a][t]["nbzero"])

# find best r2
bestR2 = 0
besti = 0

# for i in range(len(AnbZero)):
#     print(i, AnbZero[i], Ar2s[i])
# exit()

limitZero = 3

for i in range(len(Ar2s)):
    if(AnbZero[i] > limitZero or AnbZero[i]==0):
        continue

    if(min(Ar2s[i],Ar2sUVG[i])>bestR2):
        bestR2 = min(Ar2s[i],Ar2sUVG[i])
        besti = i

print("bestR2 :",bestR2)
print(besti,Aalphas[besti],Atols[besti])
print(Ar2s[besti],Ar2sUVG[besti], AnbZero[besti])

# find best mse
bestMse = 100
besti = 0
for i in range(len(Amse)):
    if(AnbZero[i] > limitZero):
        continue
    if(max(Amse[i],AmseUVG[i])<bestMse):
        bestMse = max(Amse[i],AmseUVG[i])
        besti = i

print("\n\nbestMse :",bestMse)
print(besti,Aalphas[besti],Atols[besti])
print(Amse[besti],AmseUVG[besti])
print(Ar2s[besti],Ar2sUVG[besti], AnbZero[besti])


plt.scatter(AnbZero,Ar2s,color=(0.1,0.1,0.8,0.2))
plt.scatter(AnbZero,Ar2sUVG,color=(0.8,0.1,0.8,0.2))
plt.show()

plt.scatter(AnbZero,Atols,color=(0.1,0.1,0.8,0.2))
plt.yscale("log")
plt.show()


fig = go.Figure(px.scatter_3d(log_x=True,log_y=True, title="R2-CTC and R2_UVG depending on alpha and tolerance"))
fig.add_scatter3d(name="R2_CTC",x=alphas,y=tols,z=r2s,mode="markers")
fig.add_scatter3d(name="R2_UVG",x=alphas,y=tols,z=r2sUVG,mode="markers")
fig.update_layout(scene = dict(
                    xaxis_title='Alpha',
                    yaxis_title='Tolerance',
                    zaxis_title='R2'))
fig.show()

fig = go.Figure()
fig = px.scatter_3d(log_x=True,log_y=True, title="mse-CTC and mse_UVG depending on alpha and tolerance")
fig.add_scatter3d(name="mse_CTC",x=alphas,y=tols,z=mse,mode="markers")
fig.add_scatter3d(name="mse_UVG",x=alphas,y=tols,z=mseUVG,mode="markers")
fig.update_layout(scene = dict(
                    xaxis_title='Alpha',
                    yaxis_title='Tolerance',
                    zaxis_title='MSE'))
fig.show()

fig = go.Figure()
fig = px.scatter_3d(log_x=True,log_y=True, title="number of zero coefficient depending on alpha and tolerance")
fig.add_scatter3d(name="nb_zero",x=alphas,y=tols,z=nbZero,mode="markers")
fig.update_layout(scene = dict(
                    xaxis_title='Alpha',
                    yaxis_title='Tolerance',
                    zaxis_title='nb_zero'))
fig.show()

fig = go.Figure()
fig = px.scatter_3d(log_x=False,log_y=True, title="number of zero coefficient depending on alpha and tolerance")
fig.add_scatter3d(name="r2",x=AnbZero,y=Atols,z=Ar2s,mode="markers")
fig.add_scatter3d(name="r2_UVG",x=AnbZero,y=Atols,z=Ar2sUVG,mode="markers")
fig.update_layout(scene = dict(
                    xaxis_title='NbZero',
                    yaxis_title='Tolerance',
                    zaxis_title='R2'))
fig.show()