import os
import cv2
import sys
import pandas as pd
import time
from torch.utils.data import DataLoader
# from matplotlib import pyplot as plt
# from PIL import ImageDraw, ImageFont
# from skimage import transform as tf
# from skimage.color import rgb2gray
# from skimage.transform import (AffineTransform, EuclideanTransform,
#                                ProjectiveTransform, resize, warp)
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms

from src.alignment_utils import apply_optical_flow, load_ref_frames
from src import PROJECT_DIR
from src.dataset import VDAODataset
from src.config import transformations_half

FEATURE_DIR = PROJECT_DIR + 'dataset/features/'
database_dir = '/home/luiz.tavares/Workspace/VDAO_Pixel/dataset/VDAO/'

# Num fold
if len(sys.argv) == 1:
    fold_num = 1
else:
    fold_num = int(sys.argv[2])

# =======================
window_size = 2
video_num = 2
offix = 150

# ===================================================================================================================================================
fold_type = 'test'
if fold_type == 'training':
    skip_frames = 0
    object_only = True
    min_pixels = 200
elif fold_type == 'validation':
    skip_frames = 0
    object_only = True
    min_pixels = 200
elif fold_type == 'test':
    skip_frames = 0
    object_only = False
    min_pixels = 0
else:
    skip_frames = 0
    object_only = False

# Loading dataset
alg_file = '/home/luiz.tavares/Workspace/VDAO_Pixel/dataset/alignment/'
alg_file += 'geometric_{0}.csv'.format(fold_type)
vdao_dataset = VDAODataset(fold=fold_num, split_number=0,
                           align_file=alg_file,
                           dataset_dir=database_dir, type_dataset=fold_type,
                           shuffle=False,  skip_frames=skip_frames,
                           video=video_num,
                           object_only=object_only, min_pixels=min_pixels,
                           sil_frame_ss=1,
                           transformations=transformations_half)

vdao_loader = DataLoader(vdao_dataset,
                         num_workers=1,
                         batch_size=1,
                         shuffle=False)

# Getting alignment
align_df = vdao_dataset.frames

# list reference_videos
ref_videos = align_df['reference_file'].unique()
tar_videos = align_df['target_file'].unique()

# cap
cap_refs = {i: cv2.VideoCapture(database_dir + 'ref/{0:02d}.avi'.format(i))
            for i in ref_videos}

# =======================================================================================================================================================

# previous results

res_file = '/home/luiz.tavares/Workspace/VDAO_Pixel/dataset/alignment/'
res_file += 'geometric_{0}_fold{1:02d}_fix.csv'.format(fold_type, fold_num)

if os.path.exists(res_file):
    res_df = pd.read_csv(res_file, index_col=0, low_memory=False)
    index = res_df.shape[0]
else:
    res_df = pd.DataFrame(columns=align_df.columns)
    index = 0
    os.system('touch {0}'.format(res_file))
    res_df.to_csv(res_file, header=True, encoding='utf-8')

time_0 = time.time()
for ii, (_, tar_frame, sil_frame, info) in enumerate(vdao_loader):
    print(info)

    target_file = int(info[0])
    tar_frame_idx = int(info[2])

    if ii < index:
        res_df.append(align_df.loc[ii, :])
        continue

    frame_df = align_df.loc[(align_df['test_file'] == target_file)].iloc[
        tar_frame_idx, :]
    ref_fr = frame_df['reference_frame'] + offix
    ref_file = frame_df['reference_file']
    index += 1
    max_fr = int(cap_refs[ref_file].get(cv2.CAP_PROP_FRAME_COUNT))

    ref_frames = list(range(ref_fr-window_size, ref_fr+window_size+1))
    ref_frames = [idx for idx in ref_frames if (idx >= 0 and idx <= max_fr)]
    ref_frms = load_ref_frames(cap_refs[ref_file],
                               ref_frames[0], ref_frames[-1])

    # converting target frame
    tar_frame = cv2.cvtColor(tar_frame[0].permute(1, 2, 0).numpy(),
                             cv2.COLOR_RGB2GRAY)
    sil_frame = sil_frame[0]

    rms_min, idx_ref_rms_min, model_ransac = sys.float_info.max, None, None

    for ref_frame_idx, _ in enumerate(ref_frames):
        res = apply_optical_flow(ref_frms[ref_frame_idx],
                                 tar_frame,
                                 sil_frame,
                                 bounding_box=sil_frame,
                                 apply_ransac=True,
                                 ransac_res_only=True,
                                 transformations=None)
        if res['rms_ransac_outside_bb_only'] < rms_min:
            rms_min = res['rms_ransac_outside_bb_only']
            idx_ref_rms_min = ref_frame_idx
            model_ransac = res['model_ransac']

    # Writing results
    res_df = res_df.append(align_df.loc[ii, :])
    res_df.loc[ii, 'reference_frame'] = ref_frames[ref_frame_idx]
    res_df.loc[ii, 'homography'] = ','.join([str(i)
                                             for i in
                                             model_ransac.params.reshape(-1)])
    res_df.to_csv(res_file, header=True, encoding='utf-8')

"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 1 > rf_err1.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 2 > rf_err2.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 3 > rf_err3.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 4 > rf_err4.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 5 > rf_err5.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 6 > rf_err6.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 7 > rf_err7.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 8 > rf_err8.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/src/scripts/alignment/geometric_alignment.py \
    --n_fold 9 > rf_err9.out &
"""

# Fold 1 - node-02-01
# Fold 2 - node-02-02
# Fold 3 - node-02-03
# Fold 4 - node-01-01 -> marilleva
# Fold 5 - zermatt
# Fold 6 - node-04-01
# Fold 7 - trodheim
# Fold 8 - marilleva
# Fold 9 - tampere
