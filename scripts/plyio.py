# Modified from https://antimatter15.com/splat

from plyfile import PlyData
import numpy as np
import argparse
from io import BytesIO

def ply_to_numpy(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()

    positions = np.zeros((len(sorted_indices), 3), dtype=np.float32)
    scales = np.zeros((len(sorted_indices), 3), dtype=np.float32)
    rots = np.zeros((len(sorted_indices), 4), dtype=np.float32)
    colors = np.zeros((len(sorted_indices), 4), dtype=np.float32)

    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        position = np.array([v["x"], v["y"], v["z"]], dtype=np.float32)
        scale = np.exp(
            np.array(
                [v["scale_0"], v["scale_1"], v["scale_2"]],
                dtype=np.float32,
            )
        )
        rot = np.array(
            [v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
            dtype=np.float32,
        )
        SH_C0 = 0.28209479177387814
        color = np.array(
            [
                0.5 + SH_C0 * v["f_dc_0"],
                0.5 + SH_C0 * v["f_dc_1"],
                0.5 + SH_C0 * v["f_dc_2"],
                1 / (1 + np.exp(-v["opacity"])),
            ]
        )
        positions[idx] = position
        scales[idx] = scale
        rots[idx] = rot
        colors[idx] = color

    return positions, scales, rots, colors



def ply_to_numpy_NoNormalizing(ply_file_path):
    plydata = PlyData.read(ply_file_path)
    vert = plydata["vertex"]
    sorted_indices = np.argsort(
        -np.exp(vert["scale_0"] + vert["scale_1"] + vert["scale_2"])
        / (1 + np.exp(-vert["opacity"]))
    )
    buffer = BytesIO()

    values = np.zeros((len(sorted_indices), 17), dtype=np.float32)

    for idx in sorted_indices:
        v = plydata["vertex"][idx]
        value = np.array(
                [v["x"], v["y"], v["z"],
                 v["nx"], v["ny"], v["nz"],
                 v["f_dc_0"], v["f_dc_1"], v["f_dc_2"],
                 v["opacity"],
                 v["scale_0"], v["scale_1"], v["scale_2"],
                 v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                dtype=np.float32,
            )
        
        values[idx] = value

    return values



def numpy_to_splat(positions, scales, rots, colors, output_path):
    buffer = BytesIO()
    for idx in range(len(positions)):
        position = positions[idx]
        scale = scales[idx]
        rot = rots[idx]
        color = colors[idx]
        buffer.write(position.tobytes())
        buffer.write(scale.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    splat_data = buffer.getvalue()
    with open(output_path, "wb") as f:
        f.write(splat_data)

    return splat_data



def numpy_to_splat_edited(positions, scales, rots, colors, opacity, output_path, normalized = False):

    #normalize the color values if needed
    if (not normalized):
        SH_C0 = 0.28209479177387814
        colors = colors/SH_C0-0.5
        opacity = -np.log(1/(opacity-1))
        
    colors = np.concatenate((colors, opacity), axis=1)

    buffer = BytesIO()
    for idx in range(len(positions)):
        position = positions[idx]
        scale = scales[idx]
        rot = rots[idx]
        color = colors[idx]
        buffer.write(position.tobytes())
        buffer.write(scale.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    splat_data = buffer.getvalue()
    with open(output_path, "wb") as f:
        f.write(splat_data)

    return splat_data


def main():
    parser = argparse.ArgumentParser(description="Convert PLY files to SPLAT format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input PLY files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.splat", help="The output SPLAT file."
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        positions, scales, rotations, colors = ply_to_numpy(input_file)

        numpy_to_splat(positions, scales, rotations, colors, args.output)



if __name__ == "__main__":
    main()