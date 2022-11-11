import cv2
import sys
import os
from src import PROJECT_DIR    
from src.video_dataloader import  VideoVDAODataset

fold_type = 'test'

# Extracting frames
for fold_num in range(7,10):
    
    # Loading dataset
    vdao_temporal    =  VideoVDAODataset(fold_number =fold_num, split_number=0, 
                                    align_file  = os.path.join(PROJECT_DIR,
                                                               'data/alignment/geometric_{0}_fold{1:02d}.csv'.format(fold_type, fold_num)),
                                    dataset_dir = '/nfs/proc/luiz.tavares/VDAO_Video/', 
                                    type_dataset = fold_type, shuffle = False,  skip_frames = 0,
                                    sil_frame_ss = 1,
                                    object_only=False, min_pixels=-1, transformations = [], geometric=False)
    vdao_geometric    =  VideoVDAODataset(fold_number =fold_num, split_number=0, 
                                    align_file  = os.path.join(PROJECT_DIR,
                                                               'data/alignment/geometric_{0}_fold{1:02d}.csv'.format(fold_type, fold_num)),
                                    dataset_dir = '/nfs/proc/luiz.tavares/VDAO_Video/', 
                                    type_dataset = fold_type, shuffle = False,  skip_frames = 0,
                                    sil_frame_ss = 1,
                                    object_only=False, min_pixels=-1, transformations = [], geometric=True)

    indexes = list(vdao_geometric.frames.index)
    for frame in range(len(indexes)):
        print('[{0}] {1}/{2}'.format(fold_num,frame+1, len(indexes)), end='\r')
        ref_frame, tar_frame, sil_frame, _ = vdao_temporal.__getitem__(indexes[frame])
        ref_geo_frame, _, _, _ = vdao_geometric.__getitem__(indexes[frame])
        
        cv2.imwrite('/nfs/proc/luiz.tavares/VDAO_Database/data/{2}/ref/fold{0:02d}/{1:04d}.png'.format(fold_num,indexes[frame],fold_type),
                    cv2.cvtColor(ref_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/nfs/proc/luiz.tavares/VDAO_Database/data/{2}/ref_geo/fold{0:02d}/{1:04d}.png'.format(fold_num,indexes[frame],fold_type),
                    cv2.cvtColor(ref_geo_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/nfs/proc/luiz.tavares/VDAO_Database/data/{2}/tar/fold{0:02d}/{1:04d}.png'.format(fold_num,indexes[frame],fold_type),
                    cv2.cvtColor(tar_frame, cv2.COLOR_RGB2BGR))
        cv2.imwrite('/nfs/proc/luiz.tavares/VDAO_Database/data/{2}/sil/fold{0:02d}/{1:04d}.png'.format(fold_num,indexes[frame],fold_type),
                   cv2.cvtColor(sil_frame, cv2.COLOR_RGB2GRAY ))


# conda activate pixel_env; nohup nice -n 19 python3 /home/luiz.tavares/Workspace/VDAO_Pixel/src/scripts/alignment/extract_frames.py > debug.out &