import numpy as np
import cv2
import sys
import os
import pickle
import pandas as pd

from src import RESULT_DIR
from src.utils import conf_mat

# Directories
output_dir = os.path.join(RESULT_DIR, 'test_results')
tables_dir = os.path.join(output_dir, 'tables')

# Methods
# dis_types = ['frame', 'object', 'pixel']
# gt_types  = ['bbox', 'silhouette']
# Databases
# dbases   = ['adamult', 'daomc', 'mcbs', 'mcDTSR']

# Method choices
if len(sys.argv) > 1:
    dbase = str(sys.argv[2])
else:
    dbase = 'LightGBM_warp'

# Converting results
results_csv = os.path.join(tables_dir, '{0}_full_results.csv'.format(dbase))

# Subfolders
sil_path = os.path.join(output_dir, 'silhouette')
vid_path = os.path.join(output_dir, '{0}'.format(dbase))
box_path = os.path.join(output_dir, 'bbox')


def frame_eval(vid, gt):
    gth, tn, fp, fn, tp = 0, 0, 0, 0, 0

    if 1 in gt:

        # has bounding box
        gth = 1

        # Checking if there is any detection
        if np.sum(vid) > 0:
            tp = 1
        else:
            fn = 1

    else:

        if np.sum(vid) > 0:
            fp = 1
        else:
            tn = 1

    return gth, tn, fp, fn, tp


def object_eval(vid, gt):

    # vid - predicted frame
    # gt  - ground truth

    gth, tn, fp, fn, tp = 0, 0, 0, 0, 0

    # if ground truth has an object
    if 1 in gt:

        gth = 1

        # region inside bb
        ins_region = (gt == 1).astype(np.uint8)
        out_region = (gt == 0).astype(np.uint8)

        # Checking blobs inside BB
        if (vid * ins_region).sum() > 0:
            tp = 1
        else:
            fn = 1

        # Counting blobs
        blobs_img = cv2.connectedComponents(vid.astype('uint8'))[1]

        ins_blobs = np.unique(ins_region * blobs_img)
        out_blobs = np.unique(out_region * blobs_img)

        extra_blobs = [k for k in out_blobs if k not in ins_blobs]

        # Has extra blobs
        if len(extra_blobs) > 0:
            fp = 1
        else:
            fp = 0

    # If the frame doesn't have a bounding box:
    else:

        if np.sum(vid) > 0:
            fp = 1
        else:
            tn = 1

    return gth, tn, fp, fn, tp


def pixel_eval(vid, gt):

    # Flattening arrays
    vid_arr = vid.flatten()
    gt_arr = gt.flatten()

    # have bounding box
    gth = int(np.sum(gt_arr) > 0)

    # find valid pixels
    valid_pix = np.where(gt_arr != 2)

    # confusion matrix
    tn, fp, fn, tp = conf_mat(gt_arr[valid_pix], vid_arr[valid_pix]).ravel()
    return gth, tn, fp, fn, tp


count = 0
results = {}

for vv in range(1, 60):

    slh = pickle.load(
        open(os.path.join(sil_path, '{0:02d}.pkl'.format(vv)), 'rb'))
    mtd = pickle.load(
        open(os.path.join(vid_path, '{0:02d}.pkl'.format(vv)), 'rb'))
    bbx = pickle.load(
        open(os.path.join(box_path, '{0:02d}.pkl'.format(vv)), 'rb'))

    # Getting info
    fold = mtd['fold']
    pre_ssample = mtd['pre_subsampling']
    cut_frame = mtd['cut_frame']
    post_ssample = mtd['post_subsampling']

    vid = mtd['video'].astype(np.uint8)
    sil = slh['video']
    box = bbx['video']

    sil = sil[:, ::pre_ssample, ::pre_ssample]
    h, w = sil[0].shape
    sil = sil[:, h * cut_frame // 100:(h - h * cut_frame // 100),
              w * cut_frame // 100:(w - w * cut_frame // 100)]
    box = box[:, h * cut_frame // 100:(h - h * cut_frame // 100),
              w * cut_frame // 100:(w - w * cut_frame // 100)]

    if post_ssample > 1:
        sil = np.array([
            cv2.resize(ii, (160, 90), interpolation=cv2.INTER_AREA)
            for ii in sil
        ])
        if sil.max() < 10:
            sil[sil < 0.25] = 0
            sil[sil >= 0.75] = 1
            sil[(sil >= 0.25) & (sil < 0.75)] = 2

        box = np.array([
            cv2.resize(ii, (160, 90), interpolation=cv2.INTER_AREA)
            for ii in sil
        ])
        box[box < 0.5] = 0
        box[box >= 0.5] = 1

    h, w = sil[0].shape

    for frame in range(vid.shape[0]):

        sil_frame = sil[frame]
        vid_frame = vid[frame]
        box_frame = box[frame]

        # Metrics per frame - bounding box
        gt_fb, tn_fb, fp_fb, fn_fb, tp_fb = frame_eval(vid_frame, box_frame)
        # Metrics per object - bounding box
        gt_ob, tn_ob, fp_ob, fn_ob, tp_ob = object_eval(vid_frame, box_frame)
        # Metrics per pixel - bounding box
        gt_pb, tn_pb, fp_pb, fn_pb, tp_pb = pixel_eval(vid_frame, box_frame)
        # Metrics per frame - silhouette
        gt_fs, tn_fs, fp_fs, fn_fs, tp_fs = frame_eval(vid_frame, sil_frame)
        # Metrics per object - silhouette
        gt_os, tn_os, fp_os, fn_os, tp_os = object_eval(vid_frame, sil_frame)
        # Metrics per pixel - silhouette
        gt_ps, tn_ps, fp_ps, fn_ps, tp_ps = pixel_eval(vid_frame, sil_frame)

        frame_result = {}
        frame_result['video'] = vv
        frame_result['frame'] = frame
        frame_result['fold'] = fold

        frame_result['gt_fb'] = gt_fb
        frame_result['tn_fb'] = tn_fb
        frame_result['fp_fb'] = fp_fb
        frame_result['fn_fb'] = fn_fb
        frame_result['tp_fb'] = tp_fb

        frame_result['gt_ob'] = gt_ob
        frame_result['tn_ob'] = tn_ob
        frame_result['fp_ob'] = fp_ob
        frame_result['fn_ob'] = fn_ob
        frame_result['tp_ob'] = tp_ob

        frame_result['gt_pb'] = gt_pb
        frame_result['tn_pb'] = tn_pb
        frame_result['fp_pb'] = fp_pb
        frame_result['fn_pb'] = fn_pb
        frame_result['tp_pb'] = tp_pb

        frame_result['gt_fs'] = gt_fs
        frame_result['tn_fs'] = tn_fs
        frame_result['fp_fs'] = fp_fs
        frame_result['fn_fs'] = fn_fs
        frame_result['tp_fs'] = tp_fs

        frame_result['gt_os'] = gt_os
        frame_result['tn_os'] = tn_os
        frame_result['fp_os'] = fp_os
        frame_result['fn_os'] = fn_os
        frame_result['tp_os'] = tp_os

        frame_result['gt_ps'] = gt_ps
        frame_result['tn_ps'] = tn_ps
        frame_result['fp_ps'] = fp_ps
        frame_result['fn_ps'] = fn_ps
        frame_result['tp_ps'] = tp_ps

        results[count] = frame_result
        count = count + 1
    print(vv)

df = pd.DataFrame(results).T
df.to_csv(results_csv, mode='w', header=True)
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/results/metrics_results.py \
    --dbase ADMULT > red_val.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/results/metrics_results.py \
    --dbase DAOMC   > red_val.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/results/metrics_results.py \
    --dbase MCBS_Both > red_val.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/results/metrics_results.py \
    --dbase mcDTSR  > red_val.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/results/metrics_results.py \
    --dbase 'LightGBM_warp'  > red_val.out &
"""
