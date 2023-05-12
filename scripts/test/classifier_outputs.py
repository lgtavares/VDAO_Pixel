import pickle
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import sys
import numpy as np

from src.utils import threshold, opening, closing, voting_window
from src import DATA_DIR, RESULT_DIR
from src.dataset import VDAODataset
from src.resnet import Resnet50

warnings.filterwarnings("ignore")

# Resnet50
resnet = Resnet50('cuda' if torch.cuda.is_available() else 'cpu')

# Classifier path
class_dir = os.path.join(DATA_DIR, 'test_classifier', 'classifiers')
out_dir = os.path.join(DATA_DIR, 'test_classifier', 'result_data')

# Selecting classifier
# classifiers = ['RandomForest', 'LightGBM']
# Selecting alignment
# alignments = ['temporal', 'geometric', 'warp']

save_results = True
if len(sys.argv) > 5:
    classifier = str(sys.argv[2])
    alignment = str(sys.argv[4])
    fold = int(sys.argv[6])
else:
    classifier = 'LightGBM'
    alignment = 'warp'
    fold = 1

cls_file = os.path.join(
    class_dir, '{0}_{1}_fold{2:02d}.pkl'.format(classifier, alignment, fold))
# Output file name
out_file = cls_file.replace('classifiers/', 'result_data/')

# Opening classifier file
cls = pickle.load(open(cls_file, 'rb'))

if save_results:
    save_dict_file = os.path.join(RESULT_DIR,
                                  'test_results',
                                  '{0}_{1}.pkl'.format(classifier, alignment))
    temp_dict = {}
    if not os.path.exists(save_dict_file):
        pickle.dump({}, open(save_dict_file, 'wb'))


# Training
dataset = VDAODataset(fold=fold,
                      split=-1,
                      type='test',
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
    if classifier == 'RandomForest':
        # y = cls.predict_proba(X)[:, 1].reshape((90, 160))
        y = cls.predict_proba(X)[:, 1].reshape((90, 160))
    elif classifier == 'LightGBM':
        #
        y = cls.predict(X).reshape((90, 160))
    else:
        pass
    # saving
    out_dict['prediction'][file][frame, :, :] = torch.Tensor(y)
    out_dict['silhouette'][file][frame, :, :] = sil


pickle.dump(out_dict, open(out_file, 'wb'))

if save_results:
    res_params = pickle.load(open(os.path.join(RESULT_DIR,
                             'hyperparameter_search.pkl'), 'rb'))
    res_hp = res_params['classifier']['alignment'][fold]
    num_leaves = res_hp['num_leaves']
    min_child = res_hp['min_child_samples']
    threshold_post = res_hp['threshold_post']
    opening_value = res_hp['opening']
    closing_value = res_hp['closing']
    voting_width = res_hp['voting_width']
    voting_depth = res_hp['voting_depth']
    count = res_hp['count']

    num_leaves, min_child, res_hp['threshold']

    for vv in videos:
        vid = np.array(out_dict['prediction'][vv]).astype('uint8')
        vid = threshold(vid, threshold_post)
        vid = opening(vid, opening_value)
        vid = closing(vid, closing_value)
        vid = voting_window(vid, (voting_depth,
                                  voting_width, voting_width), count / 100)
        temp_dict[int(vv)] = vid

    # Saving progress
    res_dict = pickle.load(open(save_dict_file, 'rb'))
    res_dict[int(fold)] = temp_dict
    pickle.dump(res_dict, open(save_dict_file, 'wb'))

"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 1 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 2 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 3 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 4 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 5 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 6 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 7 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 8 > lgbm_data.out &

conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/classifier_outputs.py \
    --class LightGBM --align temporal --fold 9 > lgbm_data.out &
"""
# Fold 1: node-02-01
# Fold 2: node-02-02
# Fold 3: node-02-03
# Fold 4: node-04-01
# Fold 5: cordoba -> node-02-01
# Fold 6: moscou
# Fold 7: oslo -> node-02-02
# Fold 8: taiwan
# Fold 9: leiria
