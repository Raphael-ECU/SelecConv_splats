import config

import argparse
import utils
import graph_io as gio
from mesh_helpers import loadMesh
from clusters import *
from tqdm import tqdm,trange

import matplotlib.pyplot as plt


def styletransfer(contentpath,device,out,image_type,mask,mesh):
    
    content_ref = utils.loadImage(contentpath)
    
    if mask is not None:
        mask = utils.loadMask(mask)
    
    if mesh is not None:
        mesh = loadMesh(mesh)
    
    if image_type == "2d":
        content,content_meta = gio.image2Graph(content_ref,mask=mask,depth=3,device=device)
    elif image_type == "sphere":
        content,content_meta = gio.sphere2Graph(content_ref,mask=mask,depth=3,device=device)
    elif image_type == "mesh":
        content,content_meta = gio.mesh2Graph(content_ref,mesh,depth=3,device=device)
    else:
        raise ValueError(f"image_type not known: {image_type}")

        
    # Save/show same result
    if image_type == "sphere":
        # Interpolate Point Cloud
        result_image = gio.graph2Sphere(content.x,content_meta)
    elif image_type == "mesh":
        # Interpolate Point Cloud
        result_image = gio.graph2Mesh(content.x,content_meta)
    else:
        result_image = gio.graph2Image(content.x,content_meta)

    plt.imsave(out,result_image)
    plt.imshow(result_image);plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "contentpath",
        type=str,
    )
    parser.add_argument(
        "--device",
        default= 0 if torch.cuda.is_available() else "cpu",
        choices=list(range(torch.cuda.device_count())) + ["cpu"] or ["cpu"]
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="output/output.jpg"
    )
    parser.add_argument(
        "--image_type",
        choices=("2d","panorama","cubemap","texture","superpixel","sphere","texture3D","mesh"),
        default="2d",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    styletransfer(**vars(args))


if __name__ == "__main__":
    main()
