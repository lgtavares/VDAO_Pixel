import lightgbm as lgb
import pickle
import os

from src import RESULT_DIR, DATA_DIR

classifier = 'LightGBM'
alignment = 'temporal'

for fold in range(1, 10):

    # Feature dir
    feat_dir = os.path.join(DATA_DIR, 'test_classifier', 'data')
    class_dir = os.path.join(DATA_DIR, 'test_classifier', 'classifiers')
    class_file = os.path.join(
        class_dir, '{0}_{1}_fold{2:02d}.pkl'.format(classifier, alignment,
                                                    fold))

    # Loading results
    dd = pickle.load(open(os.path.join(RESULT_DIR,
                                       'hyperparameter_search.pkl'), 'rb'))

    # Selecting columns
    dd = dd['LightGBM']['warp'][fold]

    # Storing best hyperparameters
    num_leaves = dd['num_leaves']
    num_trees = dd['num_trees']
    min_child = dd['min_child_samples']
    threshold = dd['threshold']

    # Create params dict
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting': 'goss',
        'learning_rate': 0.1,
        'lambda_l1': 1e-4,
        # 'early_stopping': 30,
        'num_booster_rounds': num_trees,
        'force_col_wise': 'true',
        'verbosity': -1,
        'num_leaves': num_leaves,
        'min_child_samples': min_child,
        'device_type': 'cpu',
        'n_jobs': 4
    }

    train_bin_file = os.path.join(
        feat_dir,
        '{0}_{1}_fold{2:02d}_trn_full.pkl'.format(classifier, alignment, fold))
    params_dataset = {'two_round': True, 'header': False, 'verbosity': -1}
    lgb_train = lgb.Dataset(data=train_bin_file,
                            params=params_dataset).construct()
    lgb_class = lgb.train(
        lgb_params,
        lgb_train,
        keep_training_booster=True,
        num_boost_round=lgb_params['num_booster_rounds'])

    pickle.dump(lgb_class, open(class_file, 'wb'))
"""
conda activate pixel_env; nohup nice -n 19 python3\
    ~/Workspace/VDAO_Pixel/scripts/test/test_classifier.py\
    > lgbm_class.out &
"""
