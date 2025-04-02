
from PIL import Image
import argparse
import os
import tqdm


parser = argparse.ArgumentParser(
                prog='LandmarkOverlay',
                description='Overlays Landmarks')
parser.add_argument('-c', '--camera', default=7, help="Defines which camera should be used.")
parser.add_argument('-n', '--head-number', default=1, help="Defines the number of the head that will be taken as input.")
args = parser.parse_args()

def main():
    print("==== START")

    for image_file in tqdm.tqdm( sorted(os.listdir(SOURCE_FOLDER)) ):
        if not image_file[-4:] == ".png":
            continue
        background = Image.open(f"{IMAGE_SOURCE_FOLDER}/frame_{image_file[:-4]}.jpg")
        foreground = Image.open(f"{SOURCE_FOLDER}/{image_file}")

        background.paste(foreground, (0, 0), foreground.convert("RGBA"))
        background.save(f"{OUTPUT_FOLDER}/{args.camera}_{image_file[:-4]}.jpg")

    print("\n==== Done!")


if __name__ == '__main__':
    SOURCE_FOLDER = "output/landmarks"
    IMAGE_SOURCE_FOLDER = f"input/MaskedImages/{str(args.camera).zfill(4)}_head_{str(args.head_number).zfill(2)}"
    OUTPUT_FOLDER = "output/heads_with_landmarks"
    main()


