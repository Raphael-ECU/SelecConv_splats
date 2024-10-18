
#needed, don't comment out
import styletransfer_mesh
import styletransfer_splat

#import scripts.pyntcloud_io as t
#import pyvista as pv



import scripts.plyio as plyio
import numpy as np

#save file as csv
scaniverse = plyio.ply_to_numpy_NoNormalizing('Scaniverse_chair_Float.ply')
np.savetxt('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_chair_Float.csv', scaniverse, delimiter=',')


#styletransfer_splat.styletransfer('Scaniverse_chair.ply', 'style_ims\style7.jpg', 'cpu', 1000)

'''
styletransfer_mesh.styletransfer('teddy', 'style_ims\style7.jpg', 'cpu', 1000)

#mesh = pv.read('C:\\Users\\Home\\Documents\\Thesis\\ply files\\Scaniverse_chair.ply')
#cpos = mesh.plot()
#plotter = pv.Plotter(off_screen=True)
#plotter.add_mesh(mesh)
#plotter.show()

m = t.read_ply('Scaniverse_chair.ply')
#m = t.read_ply('gs_Chair.ply')
#styletransfer_mesh.styletransfer('horse', 'style_ims\style2.jpg', 'cpu', 1000)
df1 = m.get('points')
#print(type(c))

#for index, row in c.iterrows():
#    print(row['x'], row['y'], row['z'])

df2 = df1[['x', 'y','z']].copy()
pointsNP = df2.to_numpy()
pv.plot(pointsNP, eye_dome_lighting=True)



points = pv.wrap(pointsNP)
surf = points.reconstruct_surface()
#surf
pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(points)
pl.add_title('Point Cloud of 3D Surface')
pl.subplot(0, 1)
pl.add_mesh(surf, color=True, show_edges=True)
pl.add_title('Reconstructed Surface')
pl.show()
'''