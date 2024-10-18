
import styletransfer_mesh
import argparse
import utils
import graph_io as gio
from mesh_helpers import loadMesh
from clusters import *
#from tqdm import tqdm,trange
import splat_helpers as splt
import clusters as cl
from torch_geometric.data import Data

from graph_networks.LinearStyleTransfer_vgg import encoder,decoder
from graph_networks.LinearStyleTransfer_matrix import TransformLayer

from graph_networks.LinearStyleTransfer.libs.Matrix import MulLayer
from graph_networks.LinearStyleTransfer.libs.models import encoder4, decoder4

#import matplotlib.pyplot as plt
from mesh_config import mesh_info



ratio=.25
depth = 3
device = 'cpu'
style_ref = utils.loadImage('style_ims\style8.jpg', shape=(256,256))

outPath ='C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Table_out_style8.splat'

pos3D, normals, colorsTorch, color_opacity, scales, rots = splt.splat_unpacker3(25, 'Table.ply',75)

up_vector = torch.tensor([[1,1,1]],dtype=torch.float)
#up_vector = 2*torch.rand((1,3))-1
up_vector = up_vector/torch.linalg.norm(up_vector,dim=1)

# Build initial graph
#edge_index are neighbors of a point, directions are the directions from that point
edge_index,directions = gh.surface2Edges(pos3D,normals,up_vector,k_neighbors=16)
#directions need to be turned into selections "W sub n" from the star-like coordinate system from Dr. Hart's github interpolated-selectionconv
edge_index,selections,interps = gh.edges2Selections(edge_index,directions,interpolated=True)

# Generate info for downsampled versions of the graph
clusters, edge_indexes, selections_list, interps_list = cl.makeSurfaceClusters(pos3D,normals,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)
#clusters, edge_indexes, selections_list, interps_list = cl.makeMeshClusters(pos3D,mesh,edge_index,selections,interps,ratio=ratio,up_vector=up_vector,depth=depth,device=device)

# Make final graph and metadata needed for mapping the result after going through the network
content = Data(x=colorsTorch,clusters=clusters,edge_indexes=edge_indexes,selections_list=selections_list,interps_list=interps_list)
content_meta = Data(pos3D=pos3D)


style,_ = gio.image2Graph(style_ref,depth=3,device=device)


# Load original network
enc_ref = encoder4()
dec_ref = decoder4()
matrix_ref = MulLayer('r41')

enc_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/vgg_r41.pth'))
dec_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/dec_r41.pth'))
matrix_ref.load_state_dict(torch.load('graph_networks/LinearStyleTransfer/models/r41.pth',map_location=torch.device(device)))

# Copy weights to graph network
enc = encoder(padding_mode="replicate")
dec = decoder(padding_mode="replicate")
matrix = TransformLayer()

with torch.no_grad():
    enc.copy_weights(enc_ref)
    dec.copy_weights(dec_ref)
    matrix.copy_weights(matrix_ref)

#content = content.to(device)
#style = style.to(device)
enc = enc.to(device)
dec = dec.to(device)
matrix = matrix.to(device)

# Run graph network
with torch.no_grad():
    cF = enc(content)
    sF = enc(style)
    feature,transmatrix = matrix(cF['r41'],sF['r41'],
                                    content.edge_indexes[3],content.selections_list[3],
                                    style.edge_indexes[3],style.selections_list[3],
                                    content.interps_list[3] if hasattr(content,'interps_list') else None)
    result = dec(feature,content)
    result = result.clamp(0,1)
    
color_opacity[:, 0:3] = result
# Save/show result
splt.splat_save(pos3D.numpy(), scales, rots, color_opacity, outPath)

