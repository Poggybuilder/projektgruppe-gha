import os
import numpy as np
import cv2
import glob
import json
import sys
from pprint import pprint as pprint
import tqdm



def CropImage(left_up, crop_size, image=None, K=None):
    crop_size = np.array(crop_size).astype(np.int32)
    left_up = np.array(left_up).astype(np.int32)
    #return image, K # BE GONE FUNCTION

    if not K is None:
        K[0:2,2] = K[0:2,2] - np.array(left_up)

    if not image is None:
        if left_up[0] < 0:
            image_left = np.zeros([image.shape[0], -left_up[0], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image_left, image])
            left_up[0] = 0
        if left_up[1] < 0:
            image_up = np.zeros([-left_up[1], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image_up, image])
            left_up[1] = 0
        if crop_size[0] + left_up[0] > image.shape[1]:
            image_right = np.zeros([image.shape[0], crop_size[0] + left_up[0] - image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.hstack([image, image_right])
        if crop_size[1] + left_up[1] > image.shape[0]:
            image_down = np.zeros([crop_size[1] + left_up[1] - image.shape[0], image.shape[1], image.shape[2]], dtype=np.uint8)
            image = np.vstack([image, image_down])

        image = image[left_up[1]:left_up[1]+crop_size[1], left_up[0]:left_up[0]+crop_size[0], :]

    return image, K


def ResizeImage(target_size, source_size, image=None, K=None):
    if not K is None:
        K[0,:] = (target_size[0] / source_size[0]) * K[0,:]
        K[1,:] = (target_size[1] / source_size[1]) * K[1,:]

    if not image is None:
        image = cv2.resize(image, dsize=target_size)
    return image, K




def extract_frames(id_list):

    for id in id_list:
        camera_path = os.path.join(DATA_SOURCE, id, 'calibration_dome.json')
        with open(camera_path, 'r') as f:
            camera_r = json.load(f)

        camera = { value["camera_id"][1:] : value for value in camera_r["cameras"]}


        UP_THREE = "\x1B[3A"

        fids = {}
        create_box()
        extend_box("Backgrounds")
        for camera_id in tqdm.tqdm(camera):
            fids[camera_id] = 0
            if not int(camera_id) in CROP_SIZE:
                continue
            left_up = LEFT_UP[int(camera_id)]
            crop_size = CROP_SIZE[int(camera_id)]

            background_path = os.path.join(DATA_SOURCE, "..", "BMV2", 'background', 'C%s.jpg' % str(int(camera_id)).zfill(4) )

            background = cv2.imread(background_path, cv2.IMREAD_COLOR)
            background, _ = ResizeImage(ORIGINAL_SIZE, None, background, None)

            background = apply_vci_rotation(background, camera_id)
            background, _ = CropImage(left_up, crop_size, background, None)
            background, _ = ResizeImage(SIZE, crop_size, background, None)

            os.makedirs(os.path.join(DATA_OUTPUT, id, 'background'), exist_ok=True)
            try:
                cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'background', 'image_' + camera_id + '.jpg'), background)
            except:
                print(f"Das Bild image_{camera_id}.jpg konnte nicht erstellt werden.\n\n")


        
        image_folders = sorted(glob.glob(os.path.join(DATA_SOURCE, id, 'frame_*')))

        count_folders = len(image_folders)
        counter = 0

        extend_box("Images")
        for image_folder in tqdm.tqdm(image_folders):
            counter += 1
            image_paths = sorted(glob.glob(os.path.join(image_folder, 'rgb', 'rgb_*.jpeg')))

            for image_path in image_paths:

                camera_id = image_path.split('/')[-1][4:-5].zfill(4)
                try:
                    left_up = LEFT_UP[int(camera_id)]
                    crop_size = CROP_SIZE[int(camera_id)]
                except:
                    left_up = [-200, 304]
                    crop_size = [2600, 2600]

                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = apply_vci_rotation(image, camera_id)

                extrinsic = np.reshape(np.array(camera[f"{camera_id}"]["extrinsics"]["view_matrix"]), (4, 4))
                extrinsic = extrinsic[:3]
                
                intrinsic = np.array(camera[f"{camera_id}"]["intrinsics"]["camera_matrix"])
                intrinsic = np.reshape(intrinsic, (3, 3))
                _, intrinsic = CropImage(left_up, crop_size, None, intrinsic)
                _, intrinsic = ResizeImage(SIZE, crop_size, None, intrinsic)
                
                visible = (np.ones_like(image) * 255).astype(np.uint8)

                image, _ = CropImage(left_up, crop_size, image, None)
                image, _ = ResizeImage(SIZE, crop_size, image, None)
                visible, _ = CropImage(left_up, crop_size, visible, None)
                visible, _ = ResizeImage(SIZE, crop_size, visible, None)
                visible = apply_vci_rotation(visible, camera_id)
                image_lowres = cv2.resize(image, SIZE_LOWRES)
                visible_lowres = cv2.resize(visible, SIZE_LOWRES)
                os.makedirs(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id]), exist_ok=True)
                cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_' + camera_id + '.jpg'), image)
                cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'image_lowres_' + camera_id + '.jpg'), image_lowres)
                cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_' + camera_id + '.jpg'), visible)
                cv2.imwrite(os.path.join(DATA_OUTPUT, id, 'images', '%04d' % fids[camera_id], 'visible_lowres_' + camera_id + '.jpg'), visible_lowres)
                os.makedirs(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id]), exist_ok=True)
                np.savez(os.path.join(DATA_OUTPUT, id, 'cameras', '%04d' % fids[camera_id], 'camera_' + camera_id + '.npz'), extrinsic=extrinsic, intrinsic=intrinsic)

                fids[camera_id] += 1
        print("\n")


def apply_vci_rotation(image, camera_id):
    if int(camera_id) in [0, 4, 5, 6, 7, 8, 12, 13, 14, 16, 18, 26, 28, 29, 34, 38]:
        result = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #print("==== Rotated 90°", end="\n")
    elif int(camera_id) >= 10000000:
        #print("==== Like VCI but actually NeRSemble")
        return image
    else:
        result = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #print("==== Rotated -90°", end="\n")
    return result


def create_box():
    print("\n")
    print(f"╒{''.join(['═' for i in range(os.get_terminal_size().columns - 4)])}═══", end="\n")

def extend_box(title):
    UP_SHIFT = "\x1B[1A"
    print(f"└── {title}:", end="\n\n")
    print(f"╘{''.join(['═' for i in range(os.get_terminal_size().columns - 4)])}═══", end = f"{UP_SHIFT}")


def init_vci_cropping():
    crop_size = {
             0: [1372, 1446], 14: [1696, 1449],   31: [1432, 1344],
             1: [1577, 1437], 16: [1459, 1200],   34: [1223, 1366],
             4: [1702, 1587], 18: [1210, 1267],   37: [1210, 1506],
             5: [1377, 1622], 20: [1322, 1173],   38: [1403, 1402],
             6: [1476, 1510], 21: [1434, 1322],   39: [1297, 1404],
             7: [1348, 1432], 24: [1472, 1378], 1000: [1272, 1450],
             8: [1462, 1533], 25: [1476, 1321], 1001: [1259, 1342],
            10: [1346, 1545], 26: [1662, 1413], 1002: [1410, 1270],
            12: [1537, 1599], 28: [1422, 1440], 1004: [1322, 1460],
            13: [1634, 1334], 29: [1400, 1397], 1005: [1301, 1380],
    }

    left_up= {
             0: [ 512,  384], 14: [ 312,  722],   31: [ 239,  534],
             1: [ 504,  296], 16: [ 367,  508],   34: [ 413,  619],
             4: [  34,  365], 18: [   0,    0],   37: [ 581,  579],
             5: [ 306,  521], 20: [ 329,  996],   38: [ 581,  631],
             6: [  22,  786], 21: [ 289,  835],   39: [ 664,  445],
             7: [ 466,  428], 24: [ 832,  254], 1000: [ 427,  571],
             8: [ 289,  565], 25: [ 417,  452], 1001: [ 374,  902],
            10: [ 623,  559], 26: [   0, 1064], 1002: [ 212, 1041],
            12: [ 483,  372], 28: [ 441,  346], 1004: [ 334,  862],
            13: [ 476,  519], 29: [ 454,  611], 1005: [ 531,  892],
            }

    return crop_size, left_up




if __name__ == "__main__":
    ORIGINAL_SIZE = [2664, 2304]
    CROP_SIZE, LEFT_UP = init_vci_cropping()
    SIZE = [2048, 2048]
    SIZE_LOWRES = [256, 256]
    DATA_SOURCE = 'VCI/preprocessing_input'
    DATA_OUTPUT = 'VCI/preprocessing_output'
    extract_frames(['head01', 'head02', 'head03', 'head04'])

