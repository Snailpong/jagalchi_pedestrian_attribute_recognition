import torch
import numpy as np
import random
import os
import argparse

DATA_DIR = "D:/해커톤-자갈치_비식별화"

def init_device_seed(seed, cuda_visible):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return device

def get_video_list():
    list_videos = []
    list_date_fol = os.listdir(DATA_DIR)

    for folder_name in list_date_fol:
        list_files = os.listdir(f"{DATA_DIR}/{folder_name}")
        list_files = sorted(set(file_name.split('.')[0] for file_name in list_files))

        for file_name in list_files:
            list_videos.append([folder_name, file_name])

    return list_videos