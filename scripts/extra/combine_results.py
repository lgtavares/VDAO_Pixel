import os
import pickle
import pandas as pd
import numpy as np
from src import RESULT_DIR
from src.utils import MCC

results = {}

for classifier in ['RandomForest', 'LightGBM']:
    results[classifier] = {}

    for alignment in ['temporal', 'geometric', 'warp']:
        results[classifier][alignment] = {}

        for fold in range(1, 10):
            results[classifier][alignment][fold] = {}

            # Result path
            class_dir = os.path.join(RESULT_DIR, classifier, alignment,
                                     'classification')
            post_dir = os.path.join(RESULT_DIR, classifier, alignment,
                                    'postprocessing')

            # Loading results
            class_file = os.path.join(class_dir,
                                      'results_fold{0:02d}.csv'.format(fold))
            post_file = os.path.join(
                post_dir, 'results_post_fold{0:02d}.csv'.format(fold))

            d_class = pd.read_csv(class_file, index_col=0)
            d_post = pd.read_csv(post_file, index_col=0)

            # Grouping by trial
            if classifier == 'LightGBM':
                d_class = d_class.groupby(['trial']).aggregate({
                    'num_trees':
                    np.average,
                    'num_leaves':
                    np.average,
                    'min_child_samples':
                    np.average,
                    'threshold':
                    np.average,
                    'tn':
                    np.sum,
                    'fp':
                    np.sum,
                    'fn':
                    np.sum,
                    'tp':
                    np.sum,
                    'time':
                    np.average
                }).reset_index()

                # Calculating MCC
                d_class['MCC'] = d_class.apply(
                    lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)

                # Maximum MCC
                dcls_max = d_class.loc[d_class['MCC'] ==
                                       d_class['MCC'].max(), :].iloc[0, :]

                res_dict = {
                    'num_leaves': int(dcls_max['num_leaves']),
                    'num_trees': int(dcls_max['num_trees']),
                    'min_child_samples': int(dcls_max['min_child_samples']),
                    'threshold': dcls_max['threshold'] / 100,
                    'MCC': dcls_max['MCC'],
                    'time': dcls_max['time']
                }

            else:
                d_class = d_class.groupby(['step']).aggregate({
                    'trees':
                    np.average,
                    'max_depth':
                    np.average,
                    'threshold':
                    np.average,
                    'tn':
                    np.sum,
                    'fp':
                    np.sum,
                    'fn':
                    np.sum,
                    'tp':
                    np.sum,
                    'time':
                    np.average
                }).reset_index()

                # Calculating MCC
                d_class['MCC'] = d_class.apply(
                    lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)

                # Maximum MCC
                dcls_max = d_class.loc[d_class['MCC'] ==
                                       d_class['MCC'].max(), :].iloc[0, :]

                res_dict = {
                    'trees': int(dcls_max['trees']),
                    'max_depth': int(dcls_max['max_depth']),
                    'threshold': dcls_max['threshold'] / 100,
                    'MCC': dcls_max['MCC'],
                    'time': dcls_max['time']
                }

            results[classifier][alignment][fold] = res_dict
            d_post = d_post.groupby(['step']).aggregate({
                'threshold': np.average,
                'voting_depth': np.average,
                'voting_width': np.average,
                'opening': np.average,
                'closing': np.average,
                'count': np.average,
                'tn': np.sum,
                'fp': np.sum,
                'fn': np.sum,
                'tp': np.sum,
                'time': np.average
            }).reset_index()

            d_post['MCC'] = d_post.apply(
                lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)
            dpost_max = d_post.loc[d_post['MCC'] ==
                                   d_post['MCC'].max(), :].iloc[0, :]

            res_df = results[classifier][alignment][fold]
            res_df['threshold_post'] = dpost_max['threshold'] / 100
            res_df['voting_depth'] = int(dpost_max['voting_depth'])
            res_df['voting_width'] = int(dpost_max['voting_width'])
            res_df['opening'] = int(dpost_max['opening'])
            res_df['closing'] = int(dpost_max['closing'])
            res_df['count'] = int(dpost_max['count'])
            res_df['time'] = dpost_max['time']
            res_df['MCC_post'] = dpost_max['MCC']
            results[classifier][alignment][fold] = res_df

pickle.dump(results,
            open(os.path.join(RESULT_DIR, 'hyperparameter_search.pkl'), 'wb'))

# Post-processing
