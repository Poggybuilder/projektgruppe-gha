import os
import torch
import argparse

from config.config import config
from lib.LandmarkDataset import LandmarkDataset
from lib.Recorder import Recorder
from lib.Fitter import Fitter
from lib.face_models import get_face_model
from lib.Camera import Camera


if __name__ == '__main__':
    print(f'\n==== Starte fitting.py')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/sample_video.yaml')
    arg = parser.parse_args()

    cfg = config()
    cfg.load(arg.config)
    cfg = cfg.get_cfg()
    
    print(f'==== Starte mit Config "{arg.config}"')

    device = torch.device('cuda:%d' % cfg.gpu_id)
    torch.cuda.set_device(cfg.gpu_id)
    
    print(f'==== CUDA Device ist:  {device}')

    dataset = LandmarkDataset(landmark_folder=cfg.landmark_folder, camera_folder=cfg.camera_folder)
    face_model = get_face_model(cfg.face_model, batch_size=len(dataset), device=device)
    camera = Camera(image_size=cfg.image_size)
    recorder = Recorder(save_folder=cfg.param_folder, visualization_folder=cfg.visualization_folder, camera=camera, visualize=cfg.visualize, save_vertices=cfg.save_vertices)

    print("==== Fitter wird gestartet\n")
    fitter = Fitter(cfg, dataset, face_model, camera, recorder, device)
    fitter.run()
