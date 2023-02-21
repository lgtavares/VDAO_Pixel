import pandas as pd
import numpy as np
import pickle
import os

from src import FEATURE_DIR, RESULT_DIR, DATA_DIR
from src.utils import MCC
from src.config import fold_split
from sklearn.ensemble import RandomForestClassifier

classifier = 'RandomForest'


for alignment in ['geometric', 'temporal', 'warp']:

    # Feature dir
    feat_dir = os.path.join(FEATURE_DIR, alignment, 'training')
    # Loading all datasets
    feat = {
        i: pd.read_csv(
            os.path.join(feat_dir, 'features_fold{0:02d}.csv'.format(i)))
        for i in range(1, 10)
    }
    [
        feat[i].insert(516, 'object', i) for i in range(1, 10)
    ]
    feat = pd.concat(feat, axis=0).reset_index(drop=True)

    for fold in range(1, 10):
        for split in range(1, 9):

            class_dir = os.path.join(DATA_DIR, 'classifiers')
            class_file = os.path.join(
                class_dir, 'rf_{0}_fold{1:02d}_spl{2:02d}.pkl'.format(
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
                'num_fold', 'step', 'num_split', 'trees', 'max_depth',
                'threshold', 'tn', 'fp', 'fn', 'tp', 'time'
            ]]

            dd.reset_index(drop=True, inplace=True)

            # Grouping by trial
            dd = dd.groupby(
                ['num_fold', 'step', 'trees', 'max_depth',
                 'threshold']).aggregate({
                     'tn': np.sum,
                     'fp': np.sum,
                     'fn': np.sum,
                     'tp': np.sum,
                     'time': np.average
                 }).reset_index()

            # Calculating MCC
            dd['MCC'] = dd.apply(
                lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)

            # Maximum MCC
            dd_max = dd.loc[dd['MCC'] == dd['MCC'].max(), :]

            # Storing best hyperparameters
            num_trees = dd_max['trees'].values[0]
            max_depth = dd_max['max_depth'].values[0]
            threshold = dd_max['threshold'].values[0] / 100

            # Splitting folds
            trn_objs = fold_split[fold][split]['training']

            # Training data
            X_train = feat.loc[feat['object'].isin(trn_objs), :].iloc[:, :512]
            y_train = feat.loc[feat['object'].isin(trn_objs), 'y']

            # Random forest classifier
            rnd_clf = RandomForestClassifier(n_estimators=num_trees,
                                             max_depth=max_depth,
                                             random_state=71,
                                             n_jobs=-1,
                                             oob_score=True)

            # Loading classifier if it is already computed
            rnd_clf.fit(X_train, y_train)

            # print(num_leaves, min_child, threshold, dd_max['MCC'].values[0])
            pickle.dump(rnd_clf, open(class_file, 'wb'))
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/classification/save_randomforest_cls.py\
    > lgbm_class.out &
"""
