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


def find_and_graph_Normals(coneSize, neighbors, fileName, displayPointCloud = False, removeTopQ = 0):
    #Get the raw PLY data into a tensor
    ply = t.read_ply(fileName)
    df1 = ply.get('points')
    df2 = df1[['x', 'y','z']].copy()
    print(df1.head(100))
    pointsNP = df2.to_numpy()
    pointsTorch = torch.Tensor(pointsNP)
    samples = pointsTorch.size()[0]
    
    y = pointsTorch
    x = pointsTorch
    assign_index = knn(x, y, neighbors)

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
    if (displayPointCloud):
        #point cloud
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

    #normals
    fig = go.Figure(data=go.Cone(
        x=pointsNP[:, 0],
        y=pointsNP[:, 2],
        z=-pointsNP[:, 1],
        u=diffvectorNum[:, 0],
        v=diffvectorNum[:, 2],
        w=-diffvectorNum[:, 1],
        sizemode="absolute",
        sizeref=coneSize,
        anchor="tail"))

    fig.update_layout(
        scene=dict(domain_x=[0, 1],
                    camera_eye=dict(x=-1.57, y=1.36, z=0.58)))

    fig.show()

    #pyvista plot points
    #pv.plot(pointsNP, eye_dome_lighting=True)

    return()




find_and_graph_Normals(4, 25, 'Scaniverse_chair.ply',displayPointCloud=True)
#find_and_graph_Normals(0.25, 25, 'Plushie.ply')
#find_and_graph_Normals(2, 25, 'Table.ply', displayPointCloud=True)
#find_and_graph_Normals(0.25, 25, 'Tower.ply')