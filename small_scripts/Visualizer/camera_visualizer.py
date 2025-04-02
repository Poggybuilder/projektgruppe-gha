
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import colormaps
import cv2
import argparse
import tqdm
import json

from pprint import pprint as pprint


parser = argparse.ArgumentParser(
    prog='CameraVisualizer',
    description='Visualizes the prositions of all cameras in cartesian space. Expects a cameras/ folder, a cameras_NeRSemble/ folder or a calibration_dome.json file'
)
parser.add_argument('-t', '--transparent', default=False, action="store_true", help="Transparent background without grid")
parser.add_argument('-d', '--dpi', default=96, help="DPI of the monitor. Necessary for matplotlib")
parser.add_argument('-r', '--full_rotation', default=False, action="store_true", help="True if the camera should rotate fully")
parser.add_argument('-s', '--size', default="2048,2048", type=str, help="Size of the image. DPI must be set correctly for this argument to work.")
use_choices = ["default", "dome", "NeRSemble"]
parser.add_argument('-u', '--use', default="default", type=str, choices=use_choices, help="Defines which kind of camera parameter format is used.")
# S ??    M 30     L 120     XL 180
dot_size_choices = {"S": 1, "M": 30, "L": 120, "XL": 180}
parser.add_argument('--dot-size', default="L", type=str, choices=dot_size_choices.keys(), help="Size of the dots representing the cameras")
args = parser.parse_args()


def main():
    print("==== START")

    
    count = 0

    try:
        size = [int(item) for item in args.size.split(',')]
        if len(size) != 2:
            print("---! For the size, please use the format '<width>,<height>'.")
            return 1
    except:
        print("---! Size could not be interpreted as two Integers")
        return 1
    size = (size[0] / args.dpi, size[1] / args.dpi)

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    
    elev = 45
    azims = [90]
    roll = 0

    if args.full_rotation:
        azims = list(np.arange(-90, 270, 1))

    
    bounds = {"X": [0,0], "Y": [0,0], "Z": [0,0]}


    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if args.use == "dome":
        frames = [i for i in range(0, 859)]
    elif args.use == "NeRSemble":
        frames = sorted(os.listdir(SOURCE_FOLDER_NERSEMBLE))
        frames = [entry for pair in zip(frames,frames,frames,frames,frames) for entry in pair]
    else:
        frames = sorted(os.listdir(SOURCE_FOLDER))



    # NOTE: Äußere Schleife
    for frame in tqdm.tqdm(frames):
        # NOTE: Background transparent
        if args.transparent:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.axis('off')
        # NOTE: Background is a grid
        else:
            ax.set_xlabel('$X$', fontsize=15, rotation=0)
            ax.set_ylabel('$Y$', fontsize=15)
            ax.set_zlabel('$Z$', fontsize=15)
            ax.tick_params("both", labelsize=7)
        ax.invert_yaxis()


        camera_data = None
        if args.use == "dome":
            with open(SOURCE_DOME, 'r') as file:
                camera_data = json.load(file)
            points = np.zeros((len(camera_data["cameras"]), 3))
            intrinsic_points = np.zeros((len(camera_data["cameras"])*INTRINSIC_POINT_COUNT, 3))
            camera_files = [f"camera{str(cam['camera_id']).zfill(4)}.npz" for cam in camera_data["cameras"] ]
        elif args.use == "NeRSemble":
            camera_files = sorted(os.listdir(os.path.join(SOURCE_FOLDER_NERSEMBLE, frame)))
            points = np.zeros((len(camera_files), 3))
            intrinsic_points = np.zeros((len(camera_files)*INTRINSIC_POINT_COUNT, 3))
        else:
            camera_files = sorted(os.listdir(os.path.join(SOURCE_FOLDER, frame)))
            points = np.zeros((len(camera_files), 3))
            intrinsic_points = np.zeros((len(camera_files)*INTRINSIC_POINT_COUNT, 3))

        colors = []
        dot_sizes = []

        # NOTE: Innere Schleife
        for idx, camera_file in enumerate(camera_files):
            if args.use == "dome":
                extrinsic = np.array(camera_data["cameras"][idx]["extrinsics"]["view_matrix"]).reshape((4,4))
                intrinsic = np.array(camera_data["cameras"][idx]["intrinsics"]["camera_matrix"]).reshape((3,3))
            elif args.use == "NeRSemble":
                position_file = os.path.join(SOURCE_FOLDER_NERSEMBLE, frame, camera_file)
                position = np.load(position_file)
                extrinsic = position["extrinsic"]
                intrinsic = position["intrinsic"]
            else:
                position_file = os.path.join(SOURCE_FOLDER, frame, camera_file)
                position = np.load(position_file)
                extrinsic = position["extrinsic"]
                intrinsic = position["intrinsic"]
            points[idx, :] = (-extrinsic[:,:3].T) @ extrinsic[:,3]

            intrinsic_points[idx * INTRINSIC_POINT_COUNT    , :] = (extrinsic[:3,:3].T) @ np.linalg.inv(intrinsic) @ [-80, -80, 0.01] + points[idx, :]
            intrinsic_points[idx * INTRINSIC_POINT_COUNT + 1, :] = (extrinsic[:3,:3].T) @ np.linalg.inv(intrinsic) @ [-80,  80, 0.01] + points[idx, :]
            intrinsic_points[idx * INTRINSIC_POINT_COUNT + 2, :] = (extrinsic[:3,:3].T) @ np.linalg.inv(intrinsic) @ [ 80, -80, 0.01] + points[idx, :]
            intrinsic_points[idx * INTRINSIC_POINT_COUNT + 3, :] = (extrinsic[:3,:3].T) @ np.linalg.inv(intrinsic) @ [ 80,  80, 0.01] + points[idx, :]


            if int(camera_file[7:-4]) in SELECTED_CAMERAS:
                colors.append("#00E868")
                dot_sizes.append(240)
            else:
                colors.append("#9B64B1")
                dot_sizes.append(dot_size)
 
        if int(frame) == 0:
            bounds["X"][0] = np.min(points[:,0])
            bounds["X"][1] = np.max(points[:,0])
            bounds["Y"][0] = np.min(points[:,1])
            bounds["Y"][1] = np.max(points[:,1])
            bounds["Z"][0] = np.min(points[:,2])
            bounds["Z"][1] = np.max(points[:,2])

        if args.full_rotation:
            elev = 180 + np.cos(np.deg2rad(int(frame))) * 90
        ax.view_init(elev, (int(frame)*360/len(frames))+0.001, 0)
        ax.scatter(
            points[:,2], points[:,0], points[:,1],
            label=f'Visualisation of Camera Positions',
            c= colors,
            s= dot_sizes
        )
        ax.scatter(
            intrinsic_points[:,2], intrinsic_points[:,0], intrinsic_points[:,1],
            label=f'Visualisation of Camera Field',
            color="#D19477",
            s= 80
        )
 
        output_folder = OUTPUT_FOLDER_NERSEMBLE if args.use == "NeRSemble" else OUTPUT_FOLDER
        plt.savefig(
            f"{output_folder}/{str(frame).zfill(4)}.jpg",
            dpi=args.dpi,
            transparent=True
        )
        plt.cla()
    print("==== Done!\n")
    print(f"==== Bounds:\n{bounds}\n==========")


if __name__ == '__main__':
    SOURCE_FOLDER           = f"input/cameras"
    SOURCE_FOLDER_NERSEMBLE = f"input/cameras_NeRSemble"
    SOURCE_DOME             = f"input/calibration_dome.json"
    OUTPUT_FOLDER           = f"output/cameras"
    OUTPUT_FOLDER_NERSEMBLE = f"output/cameras_NeRSemble"
    
    #SELECTED_CAMERAS = [0, 4, 7, 10, 12, 16, 18, 20, 25, 29, 34, 38, 39, 1001, 1002, 1005]
    SELECTED_CAMERAS = [220700191, 221501007, 222200036, 222200037, 222200038, 222200039, 222200040, 222200041]
    INTRINSIC_POINT_COUNT = 4
    
    dot_size = dot_size_choices[args.dot_size]

    main()
