import torch
import numpy as np
import graph_helpers as gh
from math import pi, sqrt, sin

def equirec2spherical(rows, cols, device = 'cpu'):
    theta_steps = torch.linspace(0, 2*pi, int(cols+1)).to(device)[:-1] # Avoid overlapping points
    phi_steps = torch.linspace(0, pi, int(rows+1)).to(device)[:-1]
    theta, phi = torch.meshgrid(theta_steps, phi_steps,indexing='xy')
    return theta.flatten(),phi.flatten()

def spherical2equirec(theta,phi,rows,cols):
    x = theta*cols/(2*pi)
    y = phi*rows/pi
    return x,y

def spherical2xyz(theta,phi):
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi),;
    return x,y,z

def sampleSphere_Equirec(rows,cols):
    theta, phi = equirec2spherical(rows, cols)
    x,y,z = spherical2xyz(theta,phi)
    
    spherical = torch.stack((theta,phi),dim=1)
    cartesian = torch.stack((x,y,z),dim=1)
    return cartesian, spherical

def sampleSphere_Layering(rows,cols=None):
    N_phi = rows

    # Make rows match the number of phi locations
    N = 4/pi*N_phi*N_phi

    # Alternative method presented in https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    a = 4*pi/N
    d = sqrt(a)
    M_phi = int(round(pi/d))
    d_phi = pi/M_phi
    d_theta = a/d_phi

    theta_list = []
    phi_list = []

    for m in range(M_phi):
        phi = pi*(m+0.5)/M_phi
        M_theta = int(round(2*pi*sin(phi)/d_theta))
        for n in range(M_theta):
            theta = 2*pi*n/M_theta

            phi_list.append(phi)
            theta_list.append(theta)

    theta = torch.tensor(theta_list,dtype=torch.float)
    phi = torch.tensor(phi_list,dtype=torch.float)
    x,y,z = spherical2xyz(theta,phi)
    
    spherical = torch.stack((theta,phi),dim=1)
    cartesian = torch.stack((x,y,z),dim=1)
    return cartesian, spherical

def sampleSphere_Spiral(rows,cols):
    # Get a sufficient number of points to represent the resolution of the image.
    N = int(2/pi*cols*rows)
    
    # Use the Fibonacci Spiral to equally space points
    # Visualization at https://www.youtube.com/watch?v=Ua0kig6N3po
    goldenRatio = (1 + 5**0.5)/2
    i = torch.arange(0, N)
    theta = (2 * pi * i / goldenRatio) % (2 * pi)
    phi = torch.arccos(1 - 2*(i+0.5)/N)
    x,y,z = spherical2xyz(theta,phi)
    
    spherical = torch.stack((theta,phi),dim=1)
    cartesian = torch.stack((x,y,z),dim=1)
    return cartesian, spherical

def sampleSphere_Icosphere(rows,cols=None):
    num_subdiv = int(np.floor(np.log2(rows)) - 1)
    cartesian = icosphere(num_subdiv)

    # Spherical coordinates calculator
    xy = cartesian[:,0]**2 + cartesian[:,1]**2
    phi = np.arctan2(np.sqrt(xy), cartesian[:,2]) # for elevation angle defined from Z-axis down
    theta = np.arctan2(cartesian[:,1], cartesian[:,0])
    theta += np.pi
    
    # Convert to torch
    phi = torch.tensor(phi,dtype=torch.float)
    theta = torch.tensor(theta,dtype=torch.float)
    
    x,y,z = spherical2xyz(theta,phi) # Redefine for axis consistency
    
    spherical = torch.stack((theta,phi),dim=1)
    cartesian = torch.stack((x,y,z),dim=1)
    return cartesian, spherical

def sampleSphere_Random(rows,cols):
    # Get a sufficient number of points to represent the resolution of the image.
    N = int(2/pi*cols*rows)

    # Alternative method presented in https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    z = 2*torch.rand(N) - 1
    phi = 2*pi*torch.rand(N)

    x = torch.sqrt(1 - z**2)*torch.cos(phi)
    y = torch.sqrt(1 - z**2)*torch.sin(phi)

    # Spherical coordinates calculator
    xy = x**2 + y**2
    phi = torch.atan2(torch.sqrt(xy), z) # for elevation angle defined from Z-axis down
    theta = torch.atan2(y, x)
    theta += pi
    
    x,y,z = spherical2xyz(theta,phi) # Redefine for axis consistency
    
    spherical = torch.stack((theta,phi),dim=1)
    cartesian = torch.stack((x,y,z),dim=1)
    return cartesian, spherical
    
#### Icosphere Methods #####
# Code taken from https://yaweiliu.github.io/research_notes/notes/20210301_Creating%20an%20icosphere%20with%20Python.html
from scipy.spatial.transform import Rotation as R

def vertex(x, y, z): 
    """ Return vertex coordinates fixed to the unit sphere """ 
    length = np.sqrt(x**2 + y**2 + z**2) 
    return [i / length for i in (x,y,z)] 

def middle_point(verts,middle_point_cache,point_1, point_2): 
    """ Find a middle point and project to the unit sphere """ 
    # We check if we have already cut this edge first 
    # to avoid duplicated verts 
    smaller_index = min(point_1, point_2) 
    greater_index = max(point_1, point_2) 
    key = '{0}-{1}'.format(smaller_index, greater_index) 
    if key in middle_point_cache: return middle_point_cache[key] 
    # If it's not in cache, then we can cut it 
    vert_1 = verts[point_1] 
    vert_2 = verts[point_2] 
    middle = [sum(i)/2 for i in zip(vert_1, vert_2)] 
    verts.append(vertex(*middle)) 
    index = len(verts) - 1 
    middle_point_cache[key] = index 
    return index

def icosphere(subdiv):
    # verts for icosahedron
    r = (1.0 + np.sqrt(5.0)) / 2.0;
    verts = np.array([[-1.0, r, 0.0],[ 1.0, r, 0.0],[-1.0, -r, 0.0],
                      [1.0, -r, 0.0],[0.0, -1.0, r],[0.0, 1.0, r],
                      [0.0, -1.0, -r],[0.0, 1.0, -r],[r, 0.0, -1.0],
                      [r, 0.0, 1.0],[ -r, 0.0, -1.0],[-r, 0.0, 1.0]]);
    # rescale the size to radius of 0.5
    verts /= np.linalg.norm(verts[0])
    # adjust the orientation
    r = R.from_quat([[0.19322862,-0.68019314,-0.19322862,0.68019314]])
    verts = r.apply(verts)
    verts = list(verts)

    faces = [[0, 11, 5],[0, 5, 1],[0, 1, 7],[0, 7, 10],
             [0, 10, 11],[1, 5, 9],[5, 11, 4],[11, 10, 2],
             [10, 7, 6],[7, 1, 8],[3, 9, 4],[3, 4, 2],
             [3, 2, 6],[3, 6, 8],[3, 8, 9],[5, 4, 9],
             [2, 4, 11],[6, 2, 10],[8, 6, 7],[9, 8, 1],];
    
    for i in range(subdiv):
        middle_point_cache = {}
        faces_subdiv = []
        for tri in faces: 
            v1  = middle_point(verts,middle_point_cache,tri[0], tri[1])
            v2  = middle_point(verts,middle_point_cache,tri[1], tri[2])
            v3  = middle_point(verts,middle_point_cache,tri[2], tri[0])
            faces_subdiv.append([tri[0], v1, v3]) 
            faces_subdiv.append([tri[1], v2, v1]) 
            faces_subdiv.append([tri[2], v3, v2]) 
            faces_subdiv.append([v1, v2, v3]) 
        faces = faces_subdiv
                
    return np.array(verts)
