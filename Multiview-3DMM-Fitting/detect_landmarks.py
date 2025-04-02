import os
import torch
import tqdm
import glob
import numpy as np
import cv2
import face_alignment
import argparse

from config.config import config

import warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/NeRSemble_031.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, flip_input=False, face_detector='blazeface', device='cuda:%d' % cfg.gpu_id)

    source_folder = cfg.image_folder
    output_folder = cfg.landmark_folder

    frames = sorted(os.listdir(source_folder))
    for frame in tqdm.tqdm(frames):
        if 'background' in frame:
            continue
        source_frame_folder = os.path.join(source_folder, frame)
        output_frame_folder = os.path.join(output_folder, frame)
        os.makedirs(output_frame_folder, exist_ok=True)

        if len(cfg.camera_ids) > 0:
            image_paths = [source_frame_folder + '/image_%s.jpg' % camera_id for camera_id in cfg.camera_ids]
            if cfg.use_masks == True:
                mask_paths = [source_frame_folder + '/mask_%s.jpg' % camera_id for camera_id in cfg.camera_ids]
        else:
            image_paths = sorted(glob.glob(source_frame_folder + '/image_*.jpg'))
            if cfg.use_masks == True:
                mask_paths = sorted(glob.glob(source_frame_folder + '/mask_*.jpg'))

        images = [cv2.resize(cv2.imread(image_path)[:, :, ::-1], (cfg.image_size, cfg.image_size)) for image_path in image_paths]
        
        if cfg.use_masks == True:
            masks = [cv2.resize(cv2.imread(mask_path)[:, :, ::-1], (cfg.image_size, cfg.image_size)) for mask_path in mask_paths]
            masked_images = np.stack([cv2.bitwise_and(image, mask) for image, mask in zip(images, masks)])
        else:
            masked_images = np.stack(images)

        #cv2.imwrite('VCI/preprocessing_masks/result.jpg', masked_images[0])
        #break

        masked_images = torch.from_numpy(masked_images).float().permute(0, 3, 1, 2).to(device)

        #print("==== Getting Landmarks")
        results = fa.get_landmarks_from_batch(masked_images, return_landmark_score=True)
        #print("==== Landmarks receive==== Landmarks receivedd")
        for i in range(len(results[0])):
            if results[1][i] is None:
                results[0][i] = np.zeros([68, 3], dtype=np.float32)
                results[1][i] = [np.zeros([68], dtype=np.float32)]
            if len(results[1][i]) > 1:
                total_score = 0.0
                for j in range(len(results[1][i])):
                    if np.sum(results[1][i][j]) > total_score:
                        total_score = np.sum(results[1][i][j])
                        landmarks_i = results[0][i][j*68:(j+1)*68]
                        scores_i = results[1][i][j:j+1]
                results[0][i] = landmarks_i
                results[1][i] = scores_i
                
        landmarks = np.concatenate([np.stack(results[0])[:, :, :2], np.stack(results[1]).transpose(0, 2, 1)], -1)
        for i, image_path in enumerate(image_paths):
            landmarks_path = os.path.join(output_frame_folder, image_path.split('/')[-1].replace('image_', 'lmk_').replace('.jpg', '.npy'))
            np.save(landmarks_path, landmarks[i])


if __name__ == '__main__':
    print("==== STARTING...")
    with warnings.catch_warnings(record=True) as active_warnings:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        
        main()

        active_warnings = filter(lambda i: issubclass(i.category, UserWarning), active_warnings)

        count = 0
        for w in active_warnings:
            count += 1
        print(f"==== There were {count} warnings:\n")
        for w in active_warnings:
            print(w.message)

    print("\n==== DONE!")




