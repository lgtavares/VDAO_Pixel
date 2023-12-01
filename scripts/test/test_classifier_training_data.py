import os
import warnings
import pandas as pd
import lightgbm as lgb
import pickle

from src import DATA_DIR, FEATURE_DIR

warnings.filterwarnings("ignore")

# Classifier path
out_dir = os.path.join(DATA_DIR, "test_classifier", "data")

params = {
    "two_round": True,
    "header": False,
    "feature_pre_filter": False,
}

classifier = "LightGBM"
alignment = "geometric"

# Training
feat_dir = os.path.join(FEATURE_DIR, alignment, "training")

for fold in range(1, 10):
    # Splitting folds
    trn_objs = [k for k in range(1, 10) if k != fold]

    # Paths
    merge_trn = os.path.join(out_dir,
                             "merge_train_fold{0:02d}.csv".format(fold))
    lgbm_trn = os.path.join(
        out_dir,
        "{0}_{1}_fold{2:02d}_trn.pkl".format(classifier, alignment, fold)
    )
    lgbm_val = lgbm_trn.replace("trn", "val")
    lgbm_trn_full = lgbm_trn.replace("trn", "trn_full")
    lgbm_tst = lgbm_trn.replace("trn", "tst")

    cmd_merge_trn = "awk '(NR == 1) || (FNR > 1)' "
    cmd_merge_trn += " ".join(
        [
            "{0}".format(os.path.join(feat_dir,
                                      "features_fold{0:02d}.csv".format(k)))
            for k in trn_objs
        ]
    ) + " > {0}".format(merge_trn)

    print(cmd_merge_trn)
    if not os.path.exists(merge_trn):
        os.system(cmd_merge_trn)

    merge_df = pd.read_csv(merge_trn)

    if classifier == "LightGBM":
        full = lgb.Dataset(
            data=merge_df.iloc[:, 0:512].values,
            label=merge_df["y"].values,
            free_raw_data=True,
            params={"two_round": True, "feature_pre_filter": False},
        )

        full = full.construct()
        full.save_binary(lgbm_trn_full)
    else:
        X_train = merge_df.iloc[:, 0:512].values
        y_train = merge_df["y"].values
        data_rf = {"X_train": X_train, "y_train": y_train}
        pickle.dump(data_rf, open(lgbm_trn_full, "wb"))

    # Training
    if not os.path.exists(lgbm_trn):
        merge_df = pd.read_csv(merge_trn)
        train_df = merge_df.sample(frac=0.8)
        val_df = merge_df.drop(train_df.index)
        train = lgb.Dataset(
            data=train_df.iloc[:, 0:512].values,
            label=train_df["y"].values,
            free_raw_data=True,
            params={"two_round": True, "feature_pre_filter": False},
        )
        val = lgb.Dataset(
            data=val_df.iloc[:, 0:512].values,
            label=val_df["y"].values,
            reference=train,
            free_raw_data=True,
            params={"two_round": True, "feature_pre_filter": False},
        )
        train = train.construct()
        train.save_binary(lgbm_trn)
        val = val.construct()
        val.save_binary(lgbm_val)
    # os.system("rm {0}".format(merge_trn))

"""
conda activate pixel_env; nohup nice -n 19 python3 \
    ~/Workspace/VDAO_Pixel/scripts/test/test_classifier_training_data.py >\
    rf_err.out &

"""
