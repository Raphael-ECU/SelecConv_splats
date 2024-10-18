
import numpy as np
import pyvista as pv



#mesh = pv.read('C:\\Users\Home\\Documents\\Thesis\\Radiance Fields Project\\Sample Datasets\\Splats\\Scaniverse_chair.ply')
mesh = pv.read('C:\\Users\\Home\\Documents\\Thesis\\IntSelcConv_with_pyVista\\Scaniverse_chair_output.ply')

cpos = mesh.plot()
