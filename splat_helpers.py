import torch
import torch.utils
import scripts.pyntcloud_io as plyio
from torch_geometric.nn import knn
import numpy as np
from torch_scatter import scatter_mean
import scripts.plyio as splatio
import pandas as pd

    

    
def splat_update_and_save(rawdata, results, fileName):


    #df1[['f_dc_0','f_dc_1','f_dc_2']] = results
    #plyNumpy = df1.to_numpy()
    #plyNumpy.tofile('Scaniverse_chair_testOut2.ply')

    #plyNumpy = df1.to_numpy()
    #np.savetxt('Scaniverse_chair_test.csv', plyNumpy, delimiter=',')

    #SH_C0 = 0.28209479177387814
    #normedResults = 0.5 - results/SH_C0

    #df1[['f_dc_0','f_dc_1','f_dc_2']] = normedResults


    #plyNumpy = df1.to_numpy()
    #np.savetxt('Scaniverse_chair_testOut.csv', plyNumpy, delimiter=',')
    #plyNumpy = df1.to_numpy()
    #plyNumpy.tofile('Scaniverse_chair_testOutNormed.ply')


    outFileName='Scaniverse_chair_outputFloat.ply'
    df1 = pd.DataFrame(rawdata, columns=['x','y','z',
                                         'nx','ny','nz',
                                         'f_dc_0','f_dc_1','f_dc_2',
                                         'opacity',
                                         'scale_0','scale_1','scale_2',
                                         'rot_0','rot_1','rot_2','rot_3'
                                         ])

    plyio.write_ply_float(filename=outFileName, points= df1, mesh=None, as_text=False)

    
    outFilePath='C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.splat'
    opacity = rawdata[:,-1].reshape(224713, 1)
    positions = rawdata[:,0:3]
    scales = rawdata[:,3:6]
    rots = rawdata[:,6:10]
    colors = results

    splatio.numpy_to_splat(positions, scales, rots, colors, opacity, outFilePath)

    #save as csv
    normalized = False
    #normalize the color values if needed
    if (normalized):
        SH_C0 = 0.28209479177387814
        colors = colors/SH_C0-0.5
        opacity = -np.log(1/(opacity-1))
        
    colors = np.concatenate((colors, opacity), axis=1)
    splat = np.concatenate((positions, scales, rots, colors), axis=1)
    #np.savetxt('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.csv', splat, delimiter=',', header=splat.dtype.names)
    np.savetxt('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_out.csv', splat, delimiter=',')

    return 

    
def splat_save(positions, scales, rots, colors, output_path):



    splatio.numpy_to_splat(positions, scales, rots, colors, output_path)

    return 






def splat_unpacker(neighbors, fileName, removeTopQ = 0):

    '''
    #Get the raw PLY data into a tensor
    plyout = plyio.read_ply('Scaniverse_chair_StyleOutput.ply')
    df1out = plyout.get('points')

    #Get the raw PLY data into a tensor
    ply = plyio.read_ply(fileName)
    df1 = ply.get('points')
    f1 = df1.to_numpy()
    f2 = df1out.to_numpy()
    '''

    positions, scales, rots, colors = splatio.ply_to_numpy(fileName)
    
    
    #Get the raw PLY data into a tensor
    torchPoints = torch.Tensor(positions)
    samples = torchPoints.size()[0]
    
    y = torchPoints
    x = torchPoints
    assign_index = knn(x, y, neighbors)

    indexSrc = assign_index[0:1][0]
    indexSrcTrn = indexSrc.reshape(-1,1)
    index = indexSrcTrn.expand(indexSrc.size()[0],3)

    src = x[assign_index[1:2]][0]
    out = src.new_zeros(src.size())
    out = scatter_mean(src, index, 0, out=out)

    diffvector = y - out[0:samples]

    diffvectorNum =diffvector.numpy()
    
    normals = torch.from_numpy(diffvectorNum)
    
    pos3D = torch.from_numpy(positions)

    torchColors = torch.Tensor(colors)
    torchColors.clamp(0,1)
    
    return pos3D, normals, colors, scales, rots




def splat_unpacker3(neighbors, fileName, removeTopQ = 100):

    '''
    #Get the raw PLY data into a tensor
    plyout = plyio.read_ply('Scaniverse_chair_StyleOutput.ply')
    df1out = plyout.get('points')

    #Get the raw PLY data into a tensor
    ply = plyio.read_ply(fileName)
    df1 = ply.get('points')
    f1 = df1.to_numpy()
    f2 = df1out.to_numpy()
    '''

    positions, scales, rots, colors = splatio.ply_to_numpy(fileName)
    
    
    #Get the raw PLY data into a tensor
    torchPoints = torch.Tensor(positions)
    samples = torchPoints.size()[0]
    
    y = torchPoints
    x = torchPoints
    assign_index = knn(x, y, neighbors)

    indexSrc = assign_index[0:1][0]
    indexSrcTrn = indexSrc.reshape(-1,1)
    index = indexSrcTrn.expand(indexSrc.size()[0],3)

    src = x[assign_index[1:2]][0]
    out = src.new_zeros(src.size())
    out = scatter_mean(src, index, 0, out=out)

    diffvector = y - out[0:samples]

    diffvectorNum =diffvector.numpy()
    
    normalsTorch = torch.from_numpy(diffvectorNum)
    
    pos3DTorch = torch.from_numpy(positions)

    colorsTorch = torch.from_numpy(colors[:,0:3])
    colorsTorch.clamp(0,1)

    dist = torch.sqrt((out[0:samples][:,0])**2 +(out[0:samples][:,1])**2+(out[0:samples][:,2])**2 )

    removeTopQ = np.clip(removeTopQ, 0, 100)
    boundry = np.percentile(dist, removeTopQ)
    mask = dist < boundry
    normalsSelect = torch.masked_select(normalsTorch, mask)
    
    return pos3DTorch, normalsTorch, colorsTorch, colors, scales, rots
