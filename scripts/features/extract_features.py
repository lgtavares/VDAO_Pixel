import os
import sys
import torch
import numpy as np
import pandas as pd
from time import time
from src import FEATURE_DIR

from src.dataset import VDAODataset
from src.resnet import Resnet50

from torch.utils.data import DataLoader

# Fold number
if len(sys.argv) == 1:
    type = 'test'
    alignment = 'temporal'
else:
    type = str(sys.argv[2])
    alignment = str(sys.argv[4])

resnet = Resnet50('cuda' if torch.cuda.is_available() else 'cpu')

delta_time = 0

for fold in range(1, 10):
    csv_file = os.path.join(FEATURE_DIR, alignment, type,
                            'features_fold{0:02d}.csv'.format(fold))
    if os.path.isfile(csv_file):
        os.system('rm ' + csv_file)

    dataset = VDAODataset(fold=fold,
                          split=0,
                          type=type,
                          alignment=alignment,
                          transform=True)
    loader = DataLoader(dataset, num_workers=1, batch_size=1, shuffle=False)

    for i_batch, (ref_frame, tar_frame, sil_frame, info) in enumerate(loader):

        time_0 = time()

        # Concatenating tensors
        feat_tar = resnet.get_features(tar_frame, 'residual3')
        feat_ref = resnet.get_features(ref_frame, 'residual3')
        feat = torch.cat((feat_tar, feat_ref), 0)

        # Reshape
        tns_full = torch.cat((feat, sil_frame[0, :, ::4, ::4]),
                             dim=0).view(513, -1).T

        # picking pixels
        if type == 'training':
            bg_pixels = np.where(tns_full[:, 512] == 0)[0]
            fg_pixels = np.where(tns_full[:, 512] == 1)[0]
            np.random.shuffle(bg_pixels)
            np.random.shuffle(fg_pixels)
            bg_pixels = bg_pixels[:72]
            fg_pixels = fg_pixels[:72]
            pixels = np.concatenate((bg_pixels, fg_pixels))
            np.random.shuffle(pixels)
        else:
            pixels = list(range(0, tns_full.shape[0]))

        time_1 = time()
        delta_time += time_1 - time_0

        # info
        ii = np.tile([int(info['file']), int(info['frame'])], (len(pixels), 1))

        # sampling
        p = np.array(pixels)[:, np.newaxis]

        feat = tns_full[pixels, :]
        full = np.hstack((feat, p, ii))
        dd = pd.DataFrame(full,
                          columns=list(range(1, 513)) +
                          ['y', 'pixel', 'file', 'frame'])

        if os.path.exists(csv_file):
            dd.to_csv(csv_file,
                      index=False,
                      header=False,
                      mode='a',
                      encoding='utf-8')
        else:
            dd.to_csv(csv_file, index=False, header=True, encoding='utf-8')

        print('[{3}] ({0}/{1})   {2:.2f}%'.format(
            i_batch + 1, len(dataset), 100 * (i_batch + 1) / len(dataset),
            fold))

# with open(os.path.join(FEATURE_DIR, 'time.txt'), 'w+') as temp_file:
#     temp_file.write('Elapsed time: {0}.\n'.format(delta_time))

# if os.path.isfile(os.path.join(FEATURE_DIR, 'debug.txt')):
#     with open(os.path.join(FEATURE_DIR, 'debug.txt'), 'w+') as temp_file:
#         temp_file.write('Features: {0}, {1}, fold {2} finished.\n'.format(
#             type, alignment, fold))
# else:
#     os.system('touch ' + os.path.join(FEATURE_DIR, 'debug.txt'))
#     with open(os.path.join(FEATURE_DIR, 'debug.txt'), 'w+') as temp_file:
#         temp_file.write('Features: {0}, {1}, fold {2} finished.\n'.format(
#             type, alignment, fold))
"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/features/extract_features.py \
        --type training --align warp > feat_debug.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/features/extract_features.py \
        --type training --align geometric > feat_debug.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/features/extract_features.py \
        --type training --align temporal > feat_debug.out &
"""
