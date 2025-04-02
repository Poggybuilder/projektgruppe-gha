
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import colormaps
import cv2
import argparse
import tqdm

from pprint import pprint as pprint


parser = argparse.ArgumentParser(
    prog='3D Landmark Visualizer',
    description='Visualizes 3D Landmarks'
)
parser.add_argument('-a', '--count', default=False, action="store_true",
                    help="Counts the number of frames in which no faces were detected.")
parser.add_argument('-t', '--transparent', default=False, action="store_true",
                    help="Removes the axes and the grid. Also makes the background transparent.")
parser.add_argument('-d', '--dpi', default=96,
                    help="Matplotlib uses the dpi-value to determine the size of the resulting images. Just enter the dpi of your screen here to receive the exact size as specified in --size.")
parser.add_argument('-n', '--head-number', default=1,
                    help="The number of the head that will be read in.")
head_model_choices = ["BFM", "FaceVerse", "FLAME", "sphere"]
parser.add_argument('-m', '--head-model', default="BFM", choices=head_model_choices,
                    help="The 3DMM-Model that was used. sphere is deprecated.")
parser.add_argument('-r', '--full_rotation', default=False, action="store_true", help="True if the camera should rotate fully")
parser.add_argument('-s', '--size', default="2048,2048", type=str, help="")
# S ??    M 30     L 120     XL 180
dot_size_choices = {"S": 1, "M": 30, "L": 120, "XL": 180}
parser.add_argument('--dot-size', default="L", type=str, choices=dot_size_choices.keys(), help="The size of the Scatter-Plot-Dots using no usual metric whatsoever.")
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

    landmark3d_file = os.path.join(SOURCE_FOLDER, f"{args.head_model}_model.npy")
    landmarks = np.load(landmark3d_file)
    landmarks = landmarks.T

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(projection="3d")
    fig.tight_layout()

    
    elev = 45
    azims = [90]
    roll = 0

    if args.full_rotation:
        azims = list(np.arange(-90, 270, 1))

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for idx, azim in tqdm.tqdm(sorted(enumerate(azims))):
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
 
        if args.full_rotation:
            elev = 45 + np.sin(np.deg2rad(idx)) * 45
        ax.view_init(elev, int(azim), roll)
        ax.scatter(
            landmarks[0], landmarks[1], landmarks[2],
            label=f'Visualisation of 3D-Landmarks',
            color ="#9B64B1",
            s= dot_size
        )
 
        plt.savefig(
            f"{OUTPUT_FOLDER}/lmk_{args.head_model}_model_{str(azim+90).zfill(4)}.jpg",
            dpi=args.dpi,
            transparent=True
        )
        plt.cla()
    print("==== Done!\n")


if __name__ == '__main__':
    SOURCE_FOLDER = f"input/landmarks3d/{str(args.head_number).zfill(2)}"
    OUTPUT_FOLDER = f"output/landmarks3d/{str(args.head_number).zfill(2)}"
    
    dot_size = dot_size_choices[args.dot_size]

    main()
