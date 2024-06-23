"""Compute FPS on Vid4 dataset.
"""
import os
import argparse
import glob
import time
import numpy as np
import torch

import models.modules.mystvsr07 as Sakuya_arch
import cv2

def read_image(path):
    """Read image from the given path.

    Args:
        path (str): The path of the image.

    Returns:
        array: RGB image.
    """
    # RGB
    img = cv2.imread(path)[:, :, ::-1]
    return img

def read_seq_images(path):
    """Read a sequence of images.

    Args:
        path (str): The path of the image sequence.

    Returns:
        array: (N, H, W, C) RGB images.
    """
    imgs_path = sorted(glob.glob(os.path.join(path, '*')))
    imgs = [read_image(img_path) for img_path in imgs_path]
    imgs = np.stack(imgs, axis=0)
    return imgs


def config_dataset():
    fps_config = {}
    fps_config['dataset_root'] = '/remote-home/cs_cs_lj/zhj/RRST/datasets/Vid4'
    fps_config['pretrain_model'] = '/remote-home/cs_cs_lj/zjj/bd_7mystvsr/experiments/bd7stvsr/models/295000_G.pth'

    return fps_config


def main():
    parser = argparse.ArgumentParser(description='Space-Time Video Super-Resolution FPS computation on Vid4 dataset')
    parser.add_argument('--config', type=str, help='Path to config file (.yaml).')
    args = parser.parse_args()

    config = config_dataset()

    LR_paths = sorted(glob.glob(os.path.join(config['dataset_root'], 'BDx4', '*')))
    imgs_LR = read_seq_images(LR_paths[0])
    imgs_LR = imgs_LR.astype(np.float32) / 255.
    imgs_LR = torch.from_numpy(imgs_LR).permute(0, 3, 1, 2).contiguous()


    scale = 4
    N_ot = 7  # 3
    N_in = 1 + N_ot // 2
    model = Sakuya_arch.LunaTokis(64, N_ot, 8, 5, 40)

    device = torch.device('cuda')
    model.load_state_dict(torch.load(config['pretrain_model']), strict=True)
    model.eval()
    model = model.to(device)

    inputs = imgs_LR[10:14].unsqueeze(0).to(device)
    torch.cuda.synchronize()
    start = time.time()

    n = 100
    for i in range(n):
        with torch.no_grad():
            outputs = model(inputs)

    torch.cuda.synchronize()
    end = time.time()
    print('fps =', n*7.0/(end-start))

if __name__ == '__main__':
    main()