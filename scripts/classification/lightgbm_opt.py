import concurrent.futures
import sys
import time
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
from skopt import gp_minimize
from skopt.callbacks import CheckpointSaver
from skopt.space import Integer
from skopt.utils import use_named_args

from src import FEATURE_DIR, RESULT_DIR
from src.utils import DIS, MCC, conf_mat, create_dir

# Hyperparameters
space = [
    Integer(8, 256, name='num_leaves'),
    Integer(5, 200, name='min_child_samples'),
    Integer(0, 100, name='threshold')
]


def get_prediction(y_hat, data):
    return 'get_prediction', y_hat, True


class LGBMClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 feature_dir='',
                 result_file='',
                 opt_file='',
                 num_fold=0,
                 metric='MCC',
                 initial_step=0,
                 random_seed=127):
        """
        Called when initializing the classifier
        """

        self.feature_dir = feature_dir
        self.num_fold = num_fold
        self.metric = metric
        self.random_seed = random_seed
        self.result_file = result_file
        self.step = initial_step

    def set_params(self, **params):
        print('====== STEP PARAMS ======')
        print(params)
        self.params = params

    def fit_model(self):

        # with Parallel(n_jobs=2) as p:
        #    self.results = p(delayed(self._run_split)(i) for i in range(1,9))
        # results = []
        # for i in range(1,9):
        #     results.append(self._run_split(i))

        # self.results = pd.DataFrame(results)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._run_split, split_num)
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

    def _run_split(self, split_num):

        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting': 'goss',
            'learning_rate': 0.1,
            'lambda_l1': 1e-4,
            'early_stopping': 30,
            'num_booster_rounds': 300,
            'force_col_wise': 'true',
            'verbosity': -1,
            'num_leaves': self.params['num_leaves'],
            'min_child_samples': self.params['min_child_samples'],
            'device_type': 'cpu',
            'n_jobs': 4
        }

        train_bin_file = os.path.join(
            self.feature_dir, 'training_fold{0:02d}_split{1:02d}.lgbdt'.format(
                self.num_fold, split_num))
        val_bin_file = os.path.join(
            self.feature_dir, 'val_fold{0:02d}_split{1:02d}.lgbdt'.format(
                self.num_fold, split_num))

        params_dataset = {
            'two_round': True,
            'header': False,
            'verbosity': -1,
            'feature_pre_filter': False
        }
        lgb_train = lgb.Dataset(data=train_bin_file,
                                params=params_dataset).construct()
        lgb_val = lgb.Dataset(val_bin_file,
                              params=params_dataset,
                              reference=lgb_train).construct()

        start_time = time.time()
        lgb_class = lgb.train(lgb_params,
                              lgb_train,
                              valid_sets=[lgb_val],
                              valid_names=['validation'],
                              keep_training_booster=True,
                              num_boost_round=lgb_params['num_booster_rounds'])

        preds = lgb_class.eval_valid(get_prediction)[1][2]
        labels = lgb_val.get_label()
        end_time = time.time()
        num_trees = lgb_class.best_iteration
        tn, fp, fn, tp = conf_mat(
            labels, preds > self.params['threshold'] / 100).ravel()
        del lgb_class, lgb_train, lgb_val

        res = {}
        res['fold_num'] = self.num_fold
        res['num_leaves'] = self.params['num_leaves']
        res['min_child_samples'] = self.params['min_child_samples']
        res['threshold'] = self.params['threshold']
        res['trial'] = self.step
        res['split_num'] = split_num
        res['num_trees'] = num_trees
        res['tn'] = tn
        res['fp'] = fp
        res['fn'] = fn
        res['tp'] = tp
        res['MCC'] = MCC(tn, fp, fn, tp)
        res['time'] = end_time - start_time

        return res


@use_named_args(space)
def objective(**params):
    lgb_class.set_params(**params)
    lgb_class.fit_model()
    return lgb_class.score()


if __name__ == "__main__":

    # Num fold
    if len(sys.argv) == 1:
        num_fold = 1
        alignment = 'warp'
    else:
        num_fold = int(sys.argv[2])
        alignment = str(sys.argv[4])

    num_steps = 300

    # Paths
    RES_OUT_DIR = os.path.join(RESULT_DIR, 'LightGBM', alignment,
                               'classification')
    create_dir(RES_OUT_DIR)
    TRN_FEAT_DIR = os.path.join(FEATURE_DIR, alignment, 'splits')

    # Important files
    result_file = os.path.join(RES_OUT_DIR,
                               'results_fold{0:02}.csv'.format(num_fold))
    opt_file = os.path.join(RES_OUT_DIR, 'opt_fold{0:02}.pkl'.format(num_fold))

    # load result file
    if os.path.exists(result_file):

        res_dd = pd.read_csv(result_file, index_col=0)
        res_dd = res_dd[[
            'fold_num', 'trial', 'split_num', 'num_leaves',
            'min_child_samples', 'threshold', 'tn', 'fp', 'fn', 'tp', 'MCC'
        ]]
        res_dd = res_dd.groupby([
            'fold_num', 'trial', 'num_leaves', 'min_child_samples', 'threshold'
        ]).aggregate({
            'tn': np.sum,
            'fp': np.sum,
            'fn': np.sum,
            'tp': np.sum
        }).reset_index()
        res_dd['MCC'] = res_dd.apply(
            lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)

        step_init = res_dd.shape[0]
        x_init = res_dd[['num_leaves', 'min_child_samples', 'threshold']]
        x_init['threshold'] = x_init['threshold'].astype('int')
        x_init = x_init.values.tolist()
        y_init = -res_dd['MCC']
        y_init = y_init.values.tolist()

        # Fold classifier
        lgb_class = LGBMClassifier(feature_dir=TRN_FEAT_DIR,
                                   result_file=result_file,
                                   opt_file=opt_file,
                                   num_fold=num_fold,
                                   initial_step=step_init,
                                   metric='MCC')

        # Optimization
        opt_saver = CheckpointSaver(opt_file, compress=9)
        res_gp = gp_minimize(objective,
                             space,
                             n_initial_points=0,
                             x0=x_init,
                             y0=y_init,
                             n_calls=num_steps - step_init,
                             callback=[opt_saver],
                             random_state=79)

    else:

        # Fold classifier
        lgb_class = LGBMClassifier(feature_dir=TRN_FEAT_DIR,
                                   result_file=result_file,
                                   opt_file=opt_file,
                                   num_fold=num_fold,
                                   metric='MCC')

        # Optimization
        opt_saver = CheckpointSaver(opt_file, compress=9)
        res_gp = gp_minimize(objective,
                             space,
                             n_calls=num_steps,
                             callback=[opt_saver],
                             random_state=79)
"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 1 --align temporal > rf_err1.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 2 --align temporal > rf_err2.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 3 --align temporal > rf_err3.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 4 --align temporal > rf_err4.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 5 --align temporal > rf_err5.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 6 --align temporal > rf_err6.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 7 --align temporal > rf_err7.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 8 --align temporal > rf_err8.out &
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/classification/lightgbm_opt.py \
        --n_fold 9 --align temporal > rf_err9.out &

MAP MACHINES
1 - node-02-01 ok
2 - node-02-02 ok
3 - node-02-03 ok
4 - tampere ok
5 - cordoba ok
6 - taiwan -> node-02-03 ok
7 - moscou  -> tampere ok
8 - oslo ok
9 - leiria ok

1 - node-02-03 ok
2 - cordoba ok
3 - oslo ok
4 - node-02-03 ok
5 - tampere ok
6 - oslo ok
7 - cordoba -> node-02-03 ok
8 - leiria -> node-02-03 ok
9 - node-04-01 ok

1 - node-02-01 ok
2 - node-02-02 ok
3 - node-02-03 ok
4 - tampere
5 - cordoba
6 - taiwan -> node-02-01
7 - moscou -> node-02-02
8 - oslo
9 - leiria-> node-02-03
"""
