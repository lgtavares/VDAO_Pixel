import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os

from src import FEATURE_DIR, RESULT_DIR, DATA_DIR
from src.utils import MCC

classifier = 'LightGBM'

for alignment in ['geometric', 'temporal', 'warp']:
    for fold in range(1, 10):
        for split in range(1, 9):

            # Feature dir
            feat_dir = os.path.join(FEATURE_DIR, alignment, 'splits')
            class_dir = os.path.join(DATA_DIR, 'classifiers')
            class_file = os.path.join(
                class_dir, 'lgbm_{0}_fold{1:02d}_spl{2:02d}.pkl'.format(
                    alignment, fold, split))

            # Result path
            res_dir = os.path.join(RESULT_DIR, classifier, alignment,
                                   'classification')
            # Loading results
            csv_file = os.path.join(res_dir,
                                    'results_fold{0:02d}.csv'.format(fold))
            dd = pd.read_csv(csv_file, index_col=0)

            # Selecting columns
            dd = dd[[
                'fold_num', 'trial', 'split_num', 'num_leaves',
                'min_child_samples', 'threshold', 'num_trees', 'tn', 'fp',
                'fn', 'tp', 'time'
            ]]

            dd.reset_index(drop=True, inplace=True)

            # Grouping by trial
            dd = dd.groupby([
                'fold_num', 'trial', 'num_leaves', 'min_child_samples',
                'threshold'
            ]).aggregate({
                'num_trees': np.average,
                'tn': np.sum,
                'fp': np.sum,
                'fn': np.sum,
                'tp': np.sum,
                'time': np.average
            }).reset_index()

            # Extra filter
            dd = dd[dd['num_leaves'] <= 200]

            # Calculating MCC
            dd['MCC'] = dd.apply(
                lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)

            # Maximum MCC
            dd_max = dd.loc[dd['MCC'] == dd['MCC'].max(), :]

            # Storing best hyperparameters
            num_leaves = dd_max['num_leaves'].values[0]
            min_child = dd_max['min_child_samples'].values[0]
            threshold = dd_max['threshold'].values[0] / 100

            # Create params dict
            lgb_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting': 'goss',
                'learning_rate': 0.1,
                'lambda_l1': 1e-4,
                'early_stopping': 30,
                'num_booster_rounds': 400,
                'force_col_wise': 'true',
                'verbosity': -1,
                'num_leaves': num_leaves,
                'min_child_samples': min_child,
                'device_type': 'cpu',
                'n_jobs': 4
            }

            suffix = 'fold{0:02d}_split{1:02d}.lgbdt'.format(fold, split)
            train_bin_file = os.path.join(feat_dir,
                                          'training_{0}'.format(suffix))

            val_bin_file = os.path.join(feat_dir, 'val_{0}'.format(suffix))

            params_dataset = {
                'two_round': True,
                'header': False,
                'verbosity': -1
            }
            lgb_train = lgb.Dataset(data=train_bin_file,
                                    params=params_dataset).construct()
            lgb_val = lgb.Dataset(val_bin_file,
                                  params=params_dataset,
                                  reference=lgb_train).construct()

            # print(num_leaves, min_child, threshold, dd_max['MCC'].values[0])

            lgb_class = lgb.train(lgb_params,
                                  lgb_train,
                                  valid_sets=[lgb_val],
                                  valid_names=['validation'],
                                  keep_training_booster=True,
                                  num_boost_round=400)

            pickle.dump(lgb_class, open(class_file, 'wb'))
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/classification/save_lightgbm_cls.py\
    > lgbm_class.out &
"""
