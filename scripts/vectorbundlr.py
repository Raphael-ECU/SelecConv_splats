import os
import styletransfer_mesh
import scripts.pyntcloud_io as t
import pandas as pd
import pyvista as pv
from pyvista import CellType
import torch
from torch_geometric.nn import knn
import matplotlib.pyplot as plt
import numpy as np
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import plotly.graph_objects as go

#src = torch.Tensor([[2, 0, 1, 4, 3], [0, 2, 1, 3, 4]])
#index = torch.tensor([[4, 5, 4, 2, 3]])
#src = torch.tensor([[4, 1, 0],[5, 1, 0],[4, 1, 0],[ 2, 1, 0],[3, 1, 0],[4, 1, 0],[5, 1, 0],[4, 1, 0],[ 2, 1, 0],[3, 1, 0]])
#index = torch.tensor([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[1,1,1]])
#out = src.new_zeros(src.size())
#print(out)
#print(src)
#out = scatter_mean(src, index, 0, out=out)

#print(out)

neighbors = 15

#Get the raw PLY data into a tensor
ply = t.read_ply('Scaniverse_chair.ply')
df1 = ply.get('points')
df2 = df1[['x', 'y','z']].copy()
pointsNP = df2.to_numpy()
pointsTorch = torch.Tensor(pointsNP)
samples = pointsTorch.size()[0]

y = pointsTorch
x = pointsTorch
assign_index = knn(x, y, neighbors)
#print('results')
#print(assign_index)
#print('Vector index')
#print(assign_index[0:1][0])
#print('Vector similarity')
#print(x[assign_index[1:2]][0])
#print(y)

indexSrc = assign_index[0:1][0]
indexSrcTrn = indexSrc.reshape(-1,1)
index = indexSrcTrn.expand(indexSrc.size()[0],3)

src = x[assign_index[1:2]][0]
out = src.new_zeros(src.size())
out = scatter_mean(src, index, 0, out=out)

#print(out[0:samples])
diff = torch.sqrt((y[:,0] - out[0:samples][:,0])**2 +(y[:,1] - out[0:samples][:,1])**2+(y[:,2] - out[0:samples][:,2])**2 )

diffvector = y - out[0:samples]

diffvectorNum =diffvector.numpy()
diffNum = diff.numpy()
resultNum =  out[0:samples].numpy()


# Normalised [0,1]
diffNumNorm = (diffNum - np.min(diffNum))/np.ptp(diffNum)
# Normalised [0,255] as integer
#norm0255Result = (255*(resultNum - np.min(resultNum))/np.ptp(resultNum)).astype(int)        
#pv.plot(pointsNP, cmap=norm0255Result)

#pv.plot(pointsNP, eye_dome_lighting=True)

#plt.plot(x[assign_index[1:2][0]])
#plt.show()




marker_data = go.Scatter3d(
    x=pointsNP[:, 0], 
    y=pointsNP[:, 2], 
    z=-pointsNP[:, 1], 
    marker=go.scatter3d.Marker(size=3, color= diffNumNorm), 
    opacity=0.8, 
    mode='markers'
)
fig=go.Figure(data=marker_data)
fig.show()


#matplotlib toooooo slow
#xs = pointsNP[:, 0]
#zs = -pointsNP[:, 1]
#ys = pointsNP[:, 2]
#fig = plt.figure(figsize=(12,7))
#ax = fig.add_subplot(projection='3d')
#img = ax.scatter(xs, ys, zs, c=diffNumNorm, cmap=plt.hot())
#fig.colorbar(img)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.show()



fig = go.Figure(data=go.Cone(
    x=pointsNP[:, 0],
    y=pointsNP[:, 2],
    z=-pointsNP[:, 1],
    u=diffvectorNum[:, 0],
    v=diffvectorNum[:, 2],
    w=-diffvectorNum[:, 1],
    sizemode="absolute",
    sizeref=4,
    anchor="tail"))

fig.update_layout(
      scene=dict(domain_x=[0, 1],
                 camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

fig.show()






fig = go.Figure(data=go.Cone(
    x=[0,1,2], y=[0,1,2], z=[0,1,2], u=[1,1,2], v=[0,1,2], w=[0,0,2],
    sizemode="absolute",
    sizeref=4,
    anchor="tail"))
