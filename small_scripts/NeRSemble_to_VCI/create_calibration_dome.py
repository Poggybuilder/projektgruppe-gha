
import json
import numpy as np
from tqdm import tqdm as tqdm
from pprint import pprint as pprint


def main():
    input_file = open(INPUT,'r')
    data = json.load(input_file)
    input_file.close()

    #pprint(data)
    output_data = {"cameras": []}

    counter = 0
    for camera_id in tqdm(data["world_2_cam"]):
        camera_dict = {}
        extrinsics0 = np.array(data["world_2_cam"][str(camera_id)][0])
        extrinsics1 = np.array(data["world_2_cam"][str(camera_id)][1])
        extrinsics2 = np.array(data["world_2_cam"][str(camera_id)][2])
        extrinsics3 = np.array(data["world_2_cam"][str(camera_id)][3])
        extrinsics  = np.concatenate((extrinsics0, extrinsics1, extrinsics2, extrinsics3), axis=0)

        intrinsics0 = np.array(data["intrinsics"][0])
        intrinsics1 = np.array(data["intrinsics"][1])
        intrinsics2 = np.array(data["intrinsics"][2])
        intrinsics  = np.concatenate((intrinsics0, intrinsics1, intrinsics2), axis=0)

        camera_dict["camera_id"] = f"C{str(camera_id).zfill(4)}"
        camera_dict["extrinsics"] = {}
        camera_dict["intrinsics"] = {}
        camera_dict["extrinsics"]["view_matrix"] = extrinsics.tolist()
        camera_dict["intrinsics"]["camera_matrix"] = intrinsics.tolist()

        output_data["cameras"].append(camera_dict)
        counter += 1

    with open(OUTPUT, 'w') as output_file:
        json.dump(output_data, output_file)



if __name__ == '__main__':
    INPUT = "camera_params/camera_params_099.json"
    OUTPUT = "output/calibration_dome.json"
    main()
