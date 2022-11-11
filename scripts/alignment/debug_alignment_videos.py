import os
import cv2
import math
import sys
import pickle
import pandas as pd
import numpy as np
import PIL.Image as Image
import torch
from matplotlib import pyplot as plt
from PIL import ImageDraw, ImageFont
from skimage import transform as tf
from skimage.color import rgb2gray
from skimage.transform import (AffineTransform, EuclideanTransform,
                               ProjectiveTransform, resize, warp)

from src import PROJECT_DIR,  EXTRA_DIR, DATA_DIR
from src.config import fold2videos, transformations_half
from src import VDAO_FRAMES_SHAPE
from src.dataset import VDAODataset

from torch.utils.data import DataLoader
from skimage.registration import optical_flow_tvl1
import torchvision.transforms as transforms


from_images = False

# ===================================================================================================================================================
fold_type = 'test'
if fold_type == 'training':
    skip_frames  = 0
    object_only  = True
    min_pixels   = 200
    frame_rate   = 24
elif fold_type == 'validation':
    skip_frames  = 0
    object_only  = True
    min_pixels   = 200
    frame_rate   = 24
elif fold_type == 'test':
    skip_frames  = 0
    object_only  = False
    min_pixels   = 0
    frame_rate   = 24
else:
    skip_frames  = 0
    object_only  = False

output_video = os.path.join(EXTRA_DIR, 'debug_align_videos')
for fold_num in range(1,10):

    # Loading dataset
    vdao_geometric = VDAODataset(fold = fold_num, split = 0, type = fold_type,
                                 alignment = 'geometric', transform = False)
    if fold_type == 'test':
        videos =  vdao_geometric.align_df.test_file.unique()
    else:
        videos =  vdao_geometric.align_df.target_file.unique()


    for video_num in videos:

        if fold_type == 'test':
            indexes = list(vdao_geometric.align_df[vdao_geometric.align_df.test_file==video_num].index)
        else:
            indexes = list(vdao_geometric.align_df[vdao_geometric.align_df.target_file==video_num].index)

        ex_image, _, _ , _ = vdao_geometric.__getitem__(0)
        h,w, _ = ex_image.shape
        out_video_name = os.path.join(output_video,
                                      '{1}_fold{2}_video{0:02}.avi'.format(int(video_num),fold_type,fold_num))
        video_out = cv2.VideoWriter(out_video_name, cv2.VideoWriter_fourcc(*'mp4v'),frame_rate,(w,h),True)

        for i, frame in enumerate(indexes):
            print(fold_num, video_num, frame, '      ', end='\r')

            if from_images:
                ref_frame = cv2.imread(os.path.join(DATA_DIR,'{2}/ref_geo/fold{0:02d}/{1:04d}.png'.format(fold_num,frame,fold_type)))
                tar_frame = cv2.imread(os.path.join(DATA_DIR,'{2}/tar/fold{0:02d}/{1:04d}.png'.format(fold_num,frame,fold_type)))
            else:
                ref_frame, tar_frame, sil_frame, _ = vdao_geometric.__getitem__(frame)


            geo_im = np.zeros((ref_frame.shape[0], ref_frame.shape[1], 3))
            geo_im[..., 0] = cv2.cvtColor(tar_frame, cv2.COLOR_RGB2GRAY)
            geo_im[..., 1] = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
            geo_im[..., 2] = cv2.cvtColor(ref_frame, cv2.COLOR_RGB2GRAY)
            geo_im = geo_im.astype('uint8')

            geo_im = cv2.putText(geo_im, '{0:<d}'.format(i), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2, cv2.LINE_AA)
            contours_in, _     = cv2.findContours((sil_frame > 0.7).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_out, _    = cv2.findContours((sil_frame > 0.3).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(geo_im, contours_in, -1, (25, 25, 201, 0.3), 1)
            cv2.drawContours(geo_im, contours_out, -1, (201, 25, 25, 0.3), 1)
            
            video_out.write(cv2.cvtColor(geo_im, cv2.COLOR_RGB2BGR))
        video_out.release()


# : conda activate pixel_env; nohup nice -n 19 python3 /home/luiz.tavares/Workspace/VDAO_Pixel/src/scripts/alignment/debug_alignment_videos.py > debug.out &