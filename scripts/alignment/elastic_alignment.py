import cv2
import sys
import numpy as np
from skimage.registration import optical_flow_tvl1
from skimage.transform import (AffineTransform, EuclideanTransform,
                               ProjectiveTransform, resize, warp)
import os
from src import PROJECT_DIR
from src.video_dataloader import VideoVDAODataset

fold_type = 'test'

# Num fold
if len(sys.argv) == 1:
    fold_num = 4
else:
    fold_num = int(sys.argv[2])
    
# Loading dataset
vdao_geometric    = VideoVDAODataset(fold_number=fold_num, split_number=0, 
                                     align_file  = os.path.join(PROJECT_DIR,
                                                               'data/alignment/geometric_{0}_fold{1:02d}.csv'.format(fold_type, fold_num)),
                                     dataset_dir = '/nfs/proc/luiz.tavares/VDAO_Video/', 
                                     type_dataset = fold_type, shuffle = False,  skip_frames = 0,
                                     object_only=False, min_pixels=0, transformations = [], geometric=False)

fold_df = vdao_geometric.frames
videos  = fold_df.target_file.unique()

for vid in videos:

    vid_df = fold_df[fold_df.target_file==vid]
    vid_index = list(vid_df.index)

    for frame in vid_index:
        if frame < 760:
            continue
        
        print(frame)
        reference_frame, target_frame, _, _ = vdao_geometric.__getitem__(frame)
        ref_frame = cv2.cvtColor(reference_frame, cv2.COLOR_RGB2GRAY)
        tar_frame = cv2.cvtColor(target_frame, cv2.COLOR_RGB2GRAY)

        # warp frame
        nr, nc = ref_frame.shape[0], ref_frame.shape[1]
        v, u = optical_flow_tvl1(tar_frame, ref_frame, tightness=0.1, num_warp=20, tol=1e-7)
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

        wref = np.zeros_like(reference_frame)
        wref[:,:,0] = 255*warp(reference_frame[:,:,0], np.array([row_coords + v, col_coords + u]), mode='edge')
        wref[:,:,1] = 255*warp(reference_frame[:,:,1], np.array([row_coords + v, col_coords + u]), mode='edge')
        wref[:,:,2] = 255*warp(reference_frame[:,:,2], np.array([row_coords + v, col_coords + u]), mode='edge')

        # cropping
        percentage = 0.05
        h, w = wref.shape[0], wref.shape[1]
        h_remove = int(h * percentage)
        w_remove = int(w * percentage)

        wref = wref[h_remove:h - h_remove, w_remove:w - w_remove,:]

        cv2.imwrite('/nfs/proc/luiz.tavares/VDAO_Database/data/{2}/ref_warp/fold{0:02d}/{1:04d}.png'.format(fold_num, frame, fold_type),
           cv2.cvtColor(wref, cv2.COLOR_RGB2BGR))

        with open('/nfs/proc/luiz.tavares/VDAO_Database/data/{0}_debug.txt'.format(fold_type), 'a') as debug_file:
            debug_file.write('{0}, {1}, {2}\n'.format(fold_num, vid ,frame))
            
# conda activate pixel_env; nohup nice -n 19 python3 /home/luiz.tavares/Workspace/VDAO_Pixel/scripts/alignment/elastic_alignment.py --fold 4 > debug.out &




# Fold 1 - node-02-01
# Fold 2 - node-02-02
# Fold 3 - node-02-03
# Fold 4 - node-01-01
# Fold 5 - cordoba
# Fold 6 - node-04-01
# Fold 7 - oslo
# Fold 8 - leiria
# Fold 9 - tampere



























# conda activate pixel_env; nohup nice -n 19 python3 /home/luiz.tavares/Workspace/VDAO_Pixel/src/scripts/alignment/extract_frames.py > debug.out &