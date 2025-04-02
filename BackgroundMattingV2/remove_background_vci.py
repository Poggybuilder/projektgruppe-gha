import argparse
import torch
import os
import shutil
import json
import glob
import cv2
#import imageio
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread

from dataset import ImagesDataset, ZipDataset
from dataset import augmentation as A
from model import MattingBase, MattingRefine
from inference_utils import HomographicAlignment



def preprocess_nersemble(args, data_folder, camera_ids):
    device = torch.device(args.device)
    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    fids = sorted(os.listdir(os.path.join(data_folder, 'images')))
    for v in range(len(camera_ids)):
        for fid in tqdm(fids):
            image_path = os.path.join(data_folder, 'images', fid, 'image_%s.jpg' % camera_ids[v])
            background_path = os.path.join(data_folder, 'background', 'image_%s.jpg' % camera_ids[v])
            if not os.path.exists(image_path):
                continue
            image = imageio.imread(image_path)
            src = (torch.from_numpy(image).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)

            if os.path.exists(background_path):
                background = imageio.imread(background_path)
                bgr = (torch.from_numpy(background).float() / 255).permute(2,0,1)[None].to(device, non_blocking=True)
            else:
                bgr = src * 0.0
                
            with torch.no_grad():
                if args.model_type == 'mattingbase':
                    pha, fgr, err, _ = model(src, bgr)
                elif args.model_type == 'mattingrefine':
                    pha, fgr, _, _, err, ref = model(src, bgr)
            mask = (pha[0].repeat([3, 1, 1]) * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            mask_lowres = cv2.resize(mask, (256, 256))

            mask_path = os.path.join(data_folder, 'images', fid, 'mask_%s.jpg' % camera_ids[v])
            imageio.imsave(mask_path, mask)

            mask_lowres_path = os.path.join(data_folder, 'images', fid, 'mask_lowres_%s.jpg' % camera_ids[v])
            imageio.imsave(mask_lowres_path, mask_lowres)


if __name__ == "__main__":
    print(f"==== STARTING...")
    parser = argparse.ArgumentParser(description='Inference images')

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--model-type', type=str, default='mattingrefine', choices=['mattingbase', 'mattingrefine'])
    parser.add_argument('--model-backbone', type=str, default='resnet101', choices=['resnet101', 'resnet50', 'mobilenetv2'])
    parser.add_argument('--model-backbone-scale', type=float, default=0.25)
    parser.add_argument('--model-checkpoint', type=str, default='model/pytorch_resnet101.pth')
    parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
    parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
    parser.add_argument('--model-refine-threshold', type=float, default=0.7)
    parser.add_argument('--model-refine-kernel-size', type=int, default=3)
    args = parser.parse_args()


    # ========================
    #  CONFIG
    # ========================
    DATA_SOURCE = 'VCI/preprocessing_output'
    HEAD_FOLDERS = ['head01', 'head02', 'head03', 'head04']

    CAMERA_IDS = ['0000', '0001', '0004', '0005', '0006', '0007', '0008', '0010', '0012', '0013',
                  '0014', '0016', '0018', '0020', '0021', '0024', '0025', '0026', '0028', '0029',
                  '0031', '0034', '0037', '0038', '0039', '1000', '1001', '1002', '1004', '1005']

    data_folders = sorted([os.path.join(DATA_SOURCE, head_folder) for head_folder in HEAD_FOLDERS])
    for data_folder in data_folders:
        print(f"==== Background Matting in {data_folder}...")
        preprocess_nersemble(args, data_folder, CAMERA_IDS)
    print(f"==== DONE!")
