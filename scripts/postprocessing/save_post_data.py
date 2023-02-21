import pickle
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import sys

from src import DATA_DIR
from src.dataset import VDAODataset
from src.resnet import Resnet50

warnings.filterwarnings("ignore")

# Resnet50
resnet = Resnet50('cuda' if torch.cuda.is_available() else 'cpu')

# Classifier path
class_dir = os.path.join(DATA_DIR, 'classifiers')

# Selecting classifier
# classifiers = ['RandomForest', 'LightGBM']
# Selecting alignment
# alignments = ['temporal', 'geometric', 'warp']

if len(sys.argv) > 2:
    classifier = str(sys.argv[2])
    alignment = str(sys.argv[4])
    fold = int(sys.argv[6])
else:
    classifier = 'RandomForest'
    alignment = 'temporal'
    fold = 3

# Selecting fold
# Selecting split
for split in tqdm(range(1, 9), desc='split'):

    # Classifier
    if classifier == 'RandomForest':
        cls_file = os.path.join(
            class_dir,
            'rf_{0}_fold{1:02d}_spl{2:02d}.pkl'.format(alignment, fold, split))
    elif classifier == 'LightGBM':
        cls_file = os.path.join(
            class_dir, 'lgbm_{0}_fold{1:02d}_spl{2:02d}.pkl'.format(
                alignment, fold, split))
    else:
        pass

    # Output file name
    out_file = cls_file.replace('classifiers/', 'post/data_')
    if os.path.exists(out_file):
        continue

    # Opening classifier file
    cls = pickle.load(open(cls_file, 'rb'))

    # Training
    dataset = VDAODataset(fold=fold,
                          split=split,
                          type='validation',
                          alignment=alignment,
                          transform=True)
    loader = DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False)

    # List of videos
    videos = dataset.align_df.test_file.unique()

    # output dictionaries
    prd_dict = {int(k): torch.zeros((201, 90, 160)) for k in videos}
    sil_dict = {int(k): torch.zeros((201, 90, 160)) for k in videos}
    out_dict = {'silhouette': sil_dict, 'prediction': prd_dict}

    for i, (ref_frame, tar_frame, sil_frame,
            info) in enumerate(tqdm(loader, desc='batch')):

        # Info
        file = int(info['test_file'])
        frame = int(info['test_frame'])

        # Concatenating tensors
        feat_tar = resnet.get_features(tar_frame, 'residual3')
        feat_ref = resnet.get_features(ref_frame, 'residual3')
        feat = torch.cat((feat_tar, feat_ref), 0)

        # Silhouettte
        sil = sil_frame[0, 0, ::4, ::4] * 255
        sil[sil < 85] = 0
        sil[(sil <= 170) & (sil >= 85)] = 0.5
        sil[sil > 170] = 1

        # Prediction
        X = feat.view(512, -1).T
        y = cls.predict_proba(X)[:, 1].reshape((90, 160))

        # saving
        out_dict['prediction'][file][frame, :, :] = torch.Tensor(y)
        out_dict['silhouette'][file][frame, :, :] = sil

    pickle.dump(out_dict, open(out_file, 'wb'))
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 1 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 2 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 3 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 4 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 5 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 6 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 7 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 8 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/save_post_data.py \
    --class RandomForest --align temporal --fold 9 > lgbm_data.out &
"""
