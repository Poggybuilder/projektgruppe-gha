
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from matplotlib import colormaps
import cv2
import argparse

from pprint import pprint as pprint


parser = argparse.ArgumentParser(
    prog='Landmark Visualizer',
    description='Reads the Landmark-Files inside the ./input/landmarks/<HeadNumber>-Folder and converts them into Images utilizing Matplotlib-Scatter-Plots.'
)
parser.add_argument('-c', '--camera', default=7,
                    help="The script visualizes a single camera for every frame.")
parser.add_argument('-a', '--count', default=False, action="store_true",
                    help="Counts the number of frames in which no faces were detected.")
parser.add_argument('-t', '--transparent', default=False, action="store_true",
                    help="Removes the axes and the grid. Also makes the background transparent.")
parser.add_argument('-d', '--dpi', default=96,
                    help="Matplotlib uses the dpi-value to determine the size of the resulting images. Just enter the dpi of your screen here to receive the exact size as specified in --size.")
parser.add_argument('-n', '--head-number', default=1,
                    help="The number of the head that will be read in. (See program description)")
parser.add_argument('-s', '--size', default="2048,2048", type=str,
                    help="The size of the resulting images.\nPlease enter as \"<width>,<height>\".\nAlso see --dpi.")
parser.add_argument('-p', '--use-params', default=False, action="store_true",
                    help="Deprecated. Do not use.")
parser.add_argument('--size-dot', default=10, type=int,
                    help="The size of the Scatter-Plot-Dots using no usual metric whatsoever.")
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

    frames = sorted(os.listdir(SOURCE_FOLDER))
    for frame in frames:
        landmark_frame_folder = os.path.join(SOURCE_FOLDER, frame)
        if args.use_params:
            landmark_file = f"{landmark_frame_folder}/lmk_3d.npy"
        else:
            landmark_file = f"{landmark_frame_folder}/lmk_{str(args.camera).zfill(4)}.npy"
        landmarks = np.load(landmark_file)
        landmarks = landmarks.T
        
        if args.count:
            if float(sum(landmarks[0])) == 0.0:
                count += 1
            print(f"\r------ Frame: {frame} \tCount: {count}", end="")
        else:
            print(f"\r------ Frame: {frame}", end="")
        
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot()
        fig.tight_layout()
        cm = colormaps['Purples']
        s2 = np.array([(args.size_dot * (2 ** (3*z))) for z in landmarks[2]])
        ax.scatter(
            landmarks[0], landmarks[1],
            label=f'Visualisation of {frame}',
            color ="#9B64B1",
        )
        

        if (args.use_params):
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            ax.set_xlim(0, size[0]*args.dpi)
            ax.set_ylim(0, size[1]*args.dpi)

        # NOTE: Background transparent
        if args.transparent:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            #ax.set_zticks([])
            plt.axis('off')

        # NOTE: Background is a grid
        else:
            ax.set_xlabel('$X$', fontsize=15, rotation=0)
            ax.set_ylabel('$Y$', fontsize=15)
            ax.tick_params("both", labelsize=7)

        ax.invert_yaxis()

        plt.savefig(
            f"output/landmarks/{frame}.png",
            dpi=args.dpi,
            transparent=True
        )
        plt.cla()
        plt.close()
    print("\n==== Done!")


if __name__ == '__main__':
    if args.use_params:
        SOURCE_FOLDER = f"input/params/{str(args.head_number).zfill(2)}"
    else:
        SOURCE_FOLDER = f"input/landmarks/{str(args.head_number).zfill(2)}"
    main()
