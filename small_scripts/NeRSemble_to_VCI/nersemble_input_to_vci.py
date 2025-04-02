
import os
import cv2
from tqdm import tqdm as tqdm


def main():
    files = os.listdir(SOURCE_FOLDER)
    camera_ids = [entry[4:-4] for entry in files]

    for idx, camera_id in enumerate(camera_ids):
        file = files[idx]

        video = cv2.VideoCapture(os.path.join(SOURCE_FOLDER, file))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        success, image = video.read()
        for frame in tqdm(range(0, frame_count)):
            os.makedirs(os.path.join(OUTPUT_FOLDER, f"frame_{str(frame).zfill(5)}", 'rgb'), exist_ok=True)
            try:
                cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"frame_{str(frame).zfill(5)}", 'rgb', 'rgb_' + camera_id + '.jpeg'), image)
            except:
                print(f"Das Bild konnte nicht erstellt werden.\n\n")
            success, image = video.read()




if __name__ == '__main__':
    ID = 99
    SOURCE_FOLDER = "input"
    OUTPUT_FOLDER = "output"
    main()
