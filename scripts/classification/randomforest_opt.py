import concurrent.futures
import sys
import time
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer
from skopt.utils import use_named_args

from src import FEATURE_DIR, RESULT_DIR
from src.utils import DIS, MCC, conf_mat, create_dir
from src.config import fold_split

# from skopt import dump, load

# Hyperparameters
space = [
    Integer(1, 200, name='trees'),
    Integer(1, 200, name='max_depth'),
    Integer(0, 100, name='threshold'),
]


class VDAORandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Random forest classifier for VDAO Dataset"""

    def __init__(self,
                 feature_dir='',
                 result_file='',
                 opt_file='',
                 num_fold=0,
                 metric='MCC',
                 random_seed=127):
        """
        Called when initializing the classifier
        """

        self.feature_dir = feature_dir
        self.num_fold = num_fold
        self.metric = metric
        self.random_seed = random_seed
        self.result_file = result_file
        self.step = 0

        # list of splits to be computed
        self.split_list = fold_split[self.num_fold]

        # Loading all datasets
        self.feat = {
            i: pd.read_csv(
                os.path.join(feature_dir,
                             'features_fold{0:02d}.csv'.format(i)))
            for i in range(1, 10) if i != num_fold
        }
        [
            self.feat[i].insert(516, 'object', i) for i in range(1, 10)
            if i != num_fold
        ]
        self.feat = pd.concat(self.feat, axis=0).reset_index(drop=True)

    def set_params(self, **params):
        print('====== STEP PARAMS ======')
        print(params)
        self.params = params

    def __fold_eval(self, num_split):

        # Counting time
        start_step = time.time()

        # Splitting folds
        trn_objs = self.split_list[num_split]['training']
        val_objs = self.split_list[num_split]['validation']

        # Training data
        X_train = self.feat.loc[
            self.feat['object'].isin(trn_objs), :].iloc[:, :512]
        y_train = self.feat.loc[self.feat['object'].isin(trn_objs), 'y']

        # Random forest classifier
        rnd_clf = RandomForestClassifier(n_estimators=self.params['trees'],
                                         max_depth=self.params['max_depth'],
                                         random_state=71,
                                         n_jobs=-1,
                                         oob_score=True)

        # Loading classifier if it is already computed
        rnd_clf.fit(X_train, y_train)

        # Validation data
        X_val = self.feat.loc[
            self.feat['object'].isin(val_objs), :].iloc[:, :512]
        y_val = self.feat.loc[self.feat['object'].isin(val_objs), 'y']

        # prediction
        prediction = rnd_clf.predict_proba(X_val)[:, 1]

        # threshold
        prediction = np.array([
            1 if v > self.params['threshold'] / 100 else 0 for v in prediction
        ],
                              dtype=float)

        # Metrics
        tn, fp, fn, tp = conf_mat(y_val, prediction).ravel()

        dis = DIS(tn, fp, fn, tp)
        mcc = MCC(tn, fp, fn, tp)
        tim = time.time() - start_step

        result = {
            'num_fold': self.num_fold,
            'step': self.step,
            'num_split': num_split,
            'trees': self.params['trees'],
            'max_depth': self.params['max_depth'],
            'threshold': self.params['threshold'],
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'DIS': dis,
            'MCC': mcc,
            'time': tim
        }

        # Finishing counting
        end_step = time.time() - start_step
        print(end_step)

        return result

    def fit_model(self):

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.__fold_eval, split_num)
                for split_num in range(1, 9)
            ]
            self.results = [f.result() for f in futures]
            self.results = pd.DataFrame(self.results)

    def _meaning(self, x):

        return (True if x >= self.params['threshold'] else False)

    def predict(self, X, y=None):
        try:
            getattr(self, "threshold_")
        except AttributeError:
            raise RuntimeError(
                "You must train classifer before predicting data!")

        return ([self._meaning(x) for x in X])

    def score(self):

        # Counting tn, fp, fn, tp
        tn_sum = self.results['tn'].sum()
        fp_sum = self.results['fp'].sum()
        fn_sum = self.results['fn'].sum()
        tp_sum = self.results['tp'].sum()

        # Computing score
        if self.metric == 'DIS':
            self.mean_res = DIS(tn_sum, fp_sum, fn_sum, tp_sum)
        elif self.metric == 'MCC':
            self.mean_res = -MCC(tn_sum, fp_sum, fn_sum, tp_sum)

        # saving results
        if self.step == 0:
            self.results.to_csv(self.result_file, mode='w', header=True)
        else:
            self.results.to_csv(self.result_file, mode='a', header=False)

        # incrementing step
        self.step += 1

        return (self.mean_res)


@use_named_args(space)
def objective(**params):
    rf_class.set_params(**params)
    rf_class.fit_model()
    return rf_class.score()


if __name__ == "__main__":

    # Num fold
    if len(sys.argv) == 1:
        num_fold = 1
    else:
        num_fold = int(sys.argv[2])

    num_steps = 300
    alignment = 'warp'

    # Paths
    RES_OUT_DIR = os.path.join(RESULT_DIR, 'RandomForest', alignment,
                               'classification')
    create_dir(RES_OUT_DIR)
    TRN_FEAT_DIR = os.path.join(FEATURE_DIR, alignment, 'training')

    # Important files
    result_file = os.path.join(RES_OUT_DIR,
                               'results_fold{0:02}.csv'.format(num_fold))
    opt_file = os.path.join(RES_OUT_DIR, 'opt_fold{0:02}.pkl'.format(num_fold))

    # Fold classifier
    rf_class = VDAORandomForestClassifier(feature_dir=TRN_FEAT_DIR,
                                          result_file=result_file,
                                          num_fold=num_fold,
                                          opt_file=opt_file,
                                          metric='MCC')

    # Optimization
    opt_saver = CheckpointSaver(opt_file, compress=9)
    res_gp = gp_minimize(objective,
                         space,
                         n_calls=num_steps,
                         callback=[opt_saver],
                         random_state=846)
"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 1 > rf_err1.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 2 > rf_err2.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 3 > rf_err3.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 4 > rf_err4.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 5 > rf_err5.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 6 > rf_err6.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 7 > rf_err7.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 8 > rf_err8.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/randomforest_opt.py \
        --n_fold 9 > rf_err9.out &

MAP MACHINES
1 - node-02-01
2 - node-02-02
3 - node-02-03
4 - tampere
5 - cordoba
6 - taiwan -> node-04-01
7 - node-01-01
8 - oslo
9 - leiria
"""
