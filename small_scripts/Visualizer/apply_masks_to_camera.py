
import sys
import tqdm
import os
import cv2
import argparse


parser = argparse.ArgumentParser(
    prog='Background Remover',
    description='Uses (pregenerated) Masks to remove the background for all frames of a single camera')
parser.add_argument('-c', '--camera', default=7, type=str, help="Camera to use")
parser.add_argument('-m', '--multiple', default=False, action="store_true", help="Betrachte mehrere Kameras")
parser.add_argument('-a', '--head-avatar', default="head_avatar", help="Head Avatar to use as Input. Folder MUST lie in ./preprocessed_input/")
parser.add_argument('-v', '--create-video', default=False, action="store_true", help="Generate a Video from the images")
parser.add_argument('--no-masks', default=False, action="store_true", help="Do not multiply with Masks but with Visibilty Indicator")
args = parser.parse_args()
    


def main():
    print("==== STARTING")
    cameras = []
    if args.multiple:
        try:
            cameras = args.camera.split(",")
        except:
            print("f---! Cameras have wrong format")
            return 1
    else:
        cameras = [args.camera]

    for camera in cameras:
        try:
            _ = int(camera)
        except:
            print(f"---! Camera {camera} must be an Integer!")
            return 1


    mask_name = "mask"
    if args.no_masks:
        mask_name = "visible"
        print("==== No Masks! Images will be created using Visibility Indicators.")


    for camera in cameras:
        print(f"==== Camera {str(camera).zfill(4)} will be used.")
        
        if not os.path.isfile(INPUT_FOLDER + "/0000/image_" + str(camera).zfill(4) + ".jpg"):
            print(f"---! Camera does not seem to exit!")
            return 1

        current_output_folder = os.path.join(OUTPUT_FOLDER, args.head_avatar)
        os.makedirs(os.path.join(current_output_folder, 'camera_' + str(camera).zfill(4)), exist_ok=True)

        if args.create_video:
            #print(f"{'visualized' if args.no_masks else 'applied_masks'}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                f"{current_output_folder}/cam_{camera}_{'visualized' if args.no_masks else 'applied_masks'}.mp4",
                fourcc,
                30,
                (2048, 2048)
            )

        frames = sorted(os.listdir(INPUT_FOLDER))
        for frame in tqdm.tqdm(frames):
            frame_path = os.path.join(INPUT_FOLDER, frame)
            image = cv2.imread(frame_path + "/image_" + str(camera).zfill(4) + ".jpg")
            mask = cv2.imread(frame_path + "/" + mask_name + "_" + str(camera).zfill(4) + ".jpg")
            
            result = cv2.bitwise_and(image, mask)
            if args.create_video:
                video_writer.write(result)
            image_path = current_output_folder + '/camera_' + str(camera).zfill(4) + "/frame_" + str(frame).zfill(4) + ".jpg"
            #print(image_path)
            cv2.imwrite(image_path, result)
        
        if args.create_video:
            #print(f"==== Writing...")
            video_writer.release()

    print("==== DONE!")




if __name__ == '__main__':
    INPUT_FOLDER = f"input/preprocessing_output/{args.head_avatar}/images"
    OUTPUT_FOLDER = "output"
    main()
