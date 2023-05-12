import sys
import os
import time
import pickle
import pandas as pd
from skopt.callbacks import CheckpointSaver
import numpy as np
from os.path import exists

from src import RESULT_DIR, DATA_DIR
from src.utils import DIS, MCC, conf_mat, create_dir
from src.config import fold_split
from src.utils import threshold, opening, closing, voting_window

from sklearn.base import BaseEstimator, ClassifierMixin
from skopt.space import Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# Hyperparameters
space = [
    Integer(1, 100, name='threshold'),
    Integer(1, 70, name='voting_depth'),
    Integer(1, 70, name='voting_width'),
    Integer(1, 70, name='opening'),
    Integer(1, 70, name='closing'),
    Integer(1, 100, name='count')
]


class PostProcEval(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 fold_num=0,
                 result_dir='',
                 video_dir='',
                 result_file='',
                 initial_step=0,
                 order='M+VW',
                 data_prefix=''):
        """
        Called when initializing the classifier
        """

        self.fold_num = fold_num
        self.result_dir = result_dir
        self.video_dir = video_dir
        self.result_file = result_file
        self.order = order
        self.step = initial_step
        self.data_prefix = data_prefix
        self.table = fold_split[self.fold_num]

        # Loading splits
        self.splits = [
            self.load_split(self.fold_num, split_num)
            for split_num in range(1, 9)
        ]

    def set_params(self, **params):
        print('====== STEP PARAMS ======')
        print(params)
        self.params = params

    def __split_eval(self, split_idx):

        spl = self.splits[split_idx]
        predictions = spl[0]
        silhouettes = spl[1]

        start_clock = time.time()

        # num_scenes = len(predictions)

        results = [
            self.scene_eval(predictions, s, self.params)
            for s in predictions.keys()
        ]

        res = np.concatenate(results, axis=0)
        sil = np.concatenate([s for s in silhouettes.values()], axis=0)

        # treating sil
        sil[sil >= 0.8] = 1
        sil[(sil >= 0.2) & (sil < 0.8)] = 2
        sil[sil < 0.2] = 0
        sil = sil.astype('uint8')

        # confusion matrix
        res_out = res.reshape(-1)
        sil_out = sil.reshape(-1)

        valid_pix = np.where(sil_out != 2)
        tn, fp, fn, tp = conf_mat(sil_out[valid_pix],
                                  res_out[valid_pix]).ravel()

        # metrics
        mcc = MCC(tn, fp, fn, tp)
        dis = DIS(tn, fp, fn, tp)

        # compute elapsed time
        elapsed_time = time.time() - start_clock

        # saving result
        result = {
            'fold': self.fold_num,
            'step': self.step,
            'threshold': self.params['threshold'],
            'voting_depth': self.params['voting_depth'],
            'voting_width': self.params['voting_width'],
            'opening': self.params['opening'],
            'closing': self.params['closing'],
            'order': self.order,
            'count': self.params['count'],
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'MCC': mcc,
            'DIS': dis,
            'time': elapsed_time
        }

        return result

    def fit(self):
        self.results = pd.DataFrame([self.__split_eval(s) for s in range(8)])

    def _meaning(self, x):
        # returns True/False according to fitted classifier
        # notice underscore on the beginning

        return (True if x >= self.params['threshold'] else False)

    def predict(self, X, y=None):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError(
                "You must train classifer before predicting data!")

        return ([self._meaning(x) for x in X])

    def score(self):

        # Computing score
        # Counting tn, fp, fn, tp
        tn_sum = self.results['tn'].sum()
        fp_sum = self.results['fp'].sum()
        fn_sum = self.results['fn'].sum()
        tp_sum = self.results['tp'].sum()

        # Computing score
        self.mean_res = -MCC(tn_sum, fp_sum, fn_sum, tp_sum)

        # saving results
        if self.step == 0:
            self.results.to_csv(self.result_file, mode='w', header=True)
        else:
            self.results.to_csv(self.result_file, mode='a', header=False)

        # incrementing step
        self.step += 1

        return (self.mean_res)

    def scene_eval(self, pred_split, scene, params):

        # parameters
        vw_width = params['voting_width']
        vw_depth = params['voting_depth']
        thresh_value = params['threshold'] / 100
        open_value = params['opening']
        close_value = params['closing']
        limit = params['count'] / 100

        # get scene
        prd_vid = pred_split[scene]
        prd_vid = threshold(prd_vid, thresh_value)

        if self.order == 'M+VW':

            prd_vid = opening(prd_vid, open_value)
            prd_vid = closing(prd_vid, close_value)
            prd_vid = voting_window(prd_vid, (vw_depth, vw_width, vw_width),
                                    limit)

        elif self.order == 'VW+M':

            prd_vid = voting_window(prd_vid, (vw_depth, vw_width, vw_width),
                                    limit)
            prd_vid = opening(prd_vid, open_value)
            prd_vid = closing(prd_vid, close_value)

        return prd_vid

    def load_split(self, fold_num, split_num):

        # Reading file
        vid_file = os.path.join(
            self.video_dir, '{0}{1:02d}.pkl'.format(self.data_prefix,
                                                    split_num))
        val_videos = pickle.load(open(vid_file, 'rb'))

        prd_vid = val_videos['prediction']
        sil_vid = val_videos['silhouette']

        return prd_vid, sil_vid


@use_named_args(space)
def objective(**params):

    pred_eval.set_params(**params)
    pred_eval.fit()
    return pred_eval.score()


if __name__ == "__main__":

    # Num fold
    if len(sys.argv) == 1:
        fold = 8
    else:
        fold = int(sys.argv[2])

    # Number of calls
    n_steps = 500

    # Experiment variables
    classifier = 'LightGBM'
    alignment = 'warp'

    # Paths
    RES_DIR = os.path.join(RESULT_DIR, classifier, alignment, 'postprocessing')
    VALIDATION_DIR = os.path.join(DATA_DIR, 'post')
    if not os.path.isdir(RES_DIR):
        create_dir(RES_DIR)

    # Important files
    opt_file = os.path.join(RES_DIR, 'opt_post_fold{0:02}.pkl'.format(fold))
    result_file = os.path.join(RES_DIR,
                               'results_post_fold{0:02}.csv'.format(fold))

    # Columns and data prefix
    if classifier == 'LightGBM':
        cc = [
            'fold_num', 'trial', 'num_leaves', 'min_child_samples', 'threshold'
        ]
        prefix = 'data_lgbm_{0}_fold{1:02d}_spl'.format(alignment, fold)
    elif classifier == 'RandomForest':
        cc = ['num_fold', 'step', 'trees', 'max_depth', 'threshold']
        prefix = 'data_rf_{0}_fold{1:02d}_spl'.format(alignment, fold)

    if exists(result_file):

        # Loading result file
        res_dd = pd.read_csv(result_file, index_col=0)
        res_dd = res_dd[[
            'fold', 'step', 'voting_depth', 'voting_width', 'threshold',
            'opening', 'closing', 'count', 'tn', 'fp', 'fn', 'tp'
        ]]
        res_dd = res_dd.groupby([
            'fold', 'step', 'voting_depth', 'voting_width', 'threshold',
            'opening', 'closing', 'count'
        ]).aggregate({
            'tn': np.sum,
            'fp': np.sum,
            'fn': np.sum,
            'tp': np.sum
        }).reset_index()
        res_dd['MCC'] = res_dd.apply(
            lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)
        step_init = res_dd.shape[0]
        x_init = res_dd[[
            'threshold', 'voting_depth', 'voting_width', 'opening', 'closing',
            'count'
        ]]
        x_init['threshold'] = x_init['threshold'].astype('int')
        x_init = x_init.values.tolist()
        y_init = -res_dd['MCC']
        y_init = y_init.values.tolist()

    else:

        # Getting hyperparameters results
        res_dir = os.path.join(RESULT_DIR, classifier, alignment,
                               'classification')

        csv_dir = os.path.join(res_dir, 'results_fold{0:02d}.csv'.format(fold))
        dd = pd.read_csv(csv_dir, index_col=0)
        dd.reset_index(drop=True, inplace=True)
        dd = dd.groupby(cc).aggregate({
            'tn': np.sum,
            'fp': np.sum,
            'fn': np.sum,
            'tp': np.sum,
            'time': np.average
        }).reset_index()
        dd['MCC'] = dd.apply(lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']),
                             axis=1)
        dd_max = dd.loc[dd['MCC'] == dd['MCC'].max(), :]

        initial_threshold = dd_max['threshold'].values[0]
        initial_point = [initial_threshold, 1, 1, 1, 1, 1]
        step_init = 0

    if exists(result_file):
        pred_eval = PostProcEval(fold_num=fold,
                                 result_dir=RES_DIR,
                                 video_dir=VALIDATION_DIR,
                                 result_file=result_file,
                                 initial_step=step_init,
                                 data_prefix=prefix)
        opt_saver = CheckpointSaver(opt_file, compress=9)
        res_gp = gp_minimize(objective,
                             space,
                             n_initial_points=0,
                             x0=x_init,
                             y0=y_init,
                             n_calls=n_steps - step_init,
                             callback=[opt_saver],
                             random_state=79)

    else:
        pred_eval = PostProcEval(fold_num=fold,
                                 result_dir=RES_DIR,
                                 video_dir=VALIDATION_DIR,
                                 result_file=result_file,
                                 data_prefix=prefix)
        opt_saver = CheckpointSaver(opt_file, compress=9)
        res_gp = gp_minimize(objective,
                             space,
                             n_initial_points=10,
                             n_calls=n_steps,
                             callback=[opt_saver],
                             n_random_starts=50,
                             x0=initial_point,
                             random_state=72,
                             noise=1e-8)
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 1 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 2 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 3 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 4 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 5 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 6 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 7 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 8 > lgbm_post_opt.out &
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/postprocessing/postprocessing_opt.py \
     --fold 9 > lgbm_post_opt.out &
"""

# Fold 1: node-02-01
# Fold 2: node-02-02
# Fold 3: node-02-03 -> moscou
# Fold 4: node-04-01
# Fold 5: node-01-01
# Fold 6: node-01-02 -> node-04-01
# Fold 7: node-01-03 -> cordoba
# Fold 8: cordoba
# Fold 9: moscou
