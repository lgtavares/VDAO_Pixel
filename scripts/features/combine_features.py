import os
import pandas as pd
import lightgbm as lgb

from src import FEATURE_DIR
from src.config import fold_split

params = {
    'two_round': True,
    'header': False,
    "feature_pre_filter": False,
}

alignment = 'warp'
feat_dir = os.path.join(FEATURE_DIR, alignment, 'training')
out_dir = os.path.join(FEATURE_DIR, alignment, 'splits')

for fold in range(1, 10):
    for split in range(1, 9):

        # Splitting folds
        split_list = fold_split[fold]
        trn_objs = split_list[split]['training']
        val_objs = split_list[split]['validation']

        # Paths
        merge_trn = os.path.join(
            out_dir,
            "merge_train_fold{0:02d}_split{1:02d}.csv".format(fold, split))
        lgbm_trn = os.path.join(
            out_dir,
            "training_fold{0:02d}_split{1:02d}.lgbdt".format(fold, split))
        merge_val = os.path.join(
            out_dir,
            "merge_val_fold{0:02d}_split{1:02d}.csv".format(fold, split))
        lgbm_val = os.path.join(
            out_dir, "val_fold{0:02d}_split{1:02d}.lgbdt".format(fold, split))

        cmd_merge_trn = "awk '(NR == 1) || (FNR > 1)' "
        cmd_merge_trn += ' '.join([
            '{0}'.format(
                os.path.join(feat_dir, 'features_fold{0:02d}.csv'.format(k)))
            for k in trn_objs
        ]) + " > {0}".format(merge_trn)

        cmd_merge_val = "awk '(NR == 1) || (FNR > 1)' "
        cmd_merge_val += ' '.join([
            '{0}'.format(
                os.path.join(feat_dir, 'features_fold{0:02d}.csv'.format(k)))
            for k in val_objs
        ]) + " > {0}".format(merge_val)

        print(cmd_merge_trn)
        if not os.path.exists(merge_trn):
            os.system(cmd_merge_trn)
        if not os.path.exists(merge_val):
            os.system(cmd_merge_val)

        # Training
        if not os.path.exists(lgbm_trn):
            merge_df = pd.read_csv(merge_trn)
            train = lgb.Dataset(data=merge_df.iloc[:, 0:512].values,
                                label=merge_df['y'].values,
                                free_raw_data=True,
                                params={
                                    'two_round': True,
                                    "feature_pre_filter": False
                                })
            train = train.construct()
            train.save_binary(lgbm_trn)

        else:
            train = lgb.Dataset(lgbm_trn, params=params).construct()

        # Validation
        if not os.path.exists(lgbm_val):
            merge_val_df = pd.read_csv(merge_val)
            val = lgb.Dataset(data=merge_val_df.iloc[:, 0:512].values,
                              label=merge_val_df['y'].values,
                              free_raw_data=True,
                              params={
                                  'two_round': True,
                                  "feature_pre_filter": False
                              },
                              reference=train)
            val = val.construct()
            val.save_binary(lgbm_val)

        os.system("rm {0} {1}".format(merge_trn, merge_val))
"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/features/combine_features.py > rf_err.out &

"""
