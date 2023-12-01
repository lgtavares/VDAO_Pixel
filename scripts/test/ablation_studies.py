import pickle
import os
import pandas as pd
import numpy as np
import warnings
import time

from src.utils import MCC, DIS, conf_mat
from src import DATA_DIR, RESULT_DIR
from src.utils import threshold, opening, closing, voting_window

warnings.filterwarnings("ignore")

save = True
out_dir = os.path.join(DATA_DIR, "test_classifier", "result_data")
result_file = os.path.join(
    RESULT_DIR, "test_results", "numerical_results", "metrics_test.pkl"
)

output_dir = os.path.join(RESULT_DIR, "ablation_studies")


def get_metrics(res, sil):
    # confusion matrix
    res_out = res.reshape(-1)
    sil_out = sil.reshape(-1).numpy()

    valid_pix = np.where(sil_out != 0.5)
    tn, fp, fn, tp = conf_mat(sil_out[valid_pix], res_out[valid_pix]).ravel()

    mcc = MCC(tn, fp, fn, tp)
    dis = DIS(tn, fp, fn, tp)

    return mcc, dis, tn, fp, fn, tp


# Ablation experiments
ablation_tests = {}
ablation_tests["normal"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["boosting"] = {
    "warping": 1,
    "boosting": 0,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["warping"] = {
    "warping": 0,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["geometric"] = {
    "warping": 2,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["opening"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 0,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["closing"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 0,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["morphology"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 0,
    "closing": 0,
    "vw_depth": 1,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["vw_depth"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 0,
    "vw_width": 1,
    "count": 1,
}
ablation_tests["vw_width"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 0,
    "count": 1,
}
ablation_tests["count"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 1,
    "vw_width": 0,
    "count": 0,
}
ablation_tests["voting_window"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 1,
    "closing": 1,
    "vw_depth": 0,
    "vw_width": 0,
    "count": 1,
}
ablation_tests["post_processing"] = {
    "warping": 1,
    "boosting": 1,
    "threshold": 1,
    "opening": 0,
    "closing": 0,
    "vw_depth": 0,
    "vw_width": 0,
    "count": 0,
}

# Experiments results
exp_results = {}

# Each experiment loop
for key, exp in ablation_tests.items():
    # Metrics dictionaries
    dict_results = {k: {} for k in range(1, 60)}

    for fold in range(1, 10):
        if exp["warping"] == 1:
            alignment = "warp"
        elif exp["warping"] == 0:
            alignment = "temporal"
        else:
            alignment = "geometric"

        if exp["boosting"]:
            classifier = "LightGBM"
        else:
            classifier = "RandomForest"

        # Prediction file
        res_file = os.path.join(
            out_dir,
            "{2}_{0}_fold{1:02d}.pkl".format(alignment, fold, classifier)
        )
        data = pickle.load(open(res_file, "rb"))

        # Loading results
        res_clf_dir = os.path.join(RESULT_DIR, classifier,
                                   alignment, "classification")
        res_pst_dir = os.path.join(RESULT_DIR, classifier,
                                   alignment, "postprocessing")

        clf_csv = os.path.join(res_clf_dir,
                               "results_fold{0:02d}.csv".format(fold))
        post_csv = os.path.join(
            res_pst_dir, "results_post_fold{0:02d}.csv".format(fold)
        )

        dd_clf = pd.read_csv(clf_csv, index_col=0)
        dd_post = pd.read_csv(post_csv, index_col=0)
        dd_clf.reset_index(drop=True, inplace=True)
        dd_post.reset_index(drop=True, inplace=True)

        # Grouping by trial
        if exp["boosting"]:
            clf_cols = ['fold_num', 'trial', 'num_leaves',
                        'min_child_samples', 'threshold']
        else:
            clf_cols = ['num_fold', 'step', 'trees', 'max_depth', 'threshold']

        dd_clf = (
            dd_clf.groupby(
                clf_cols
            )
            .aggregate(
                {
                    "tn": np.sum,
                    "fp": np.sum,
                    "fn": np.sum,
                    "tp": np.sum,
                    "time": np.average,
                }
            )
            .reset_index()
        )
        dd_post = (
            dd_post.groupby(
                [
                    "fold",
                    "step",
                    "threshold",
                    "voting_depth",
                    "voting_width",
                    "opening",
                    "closing",
                    "count",
                ]
            )
            .aggregate(
                {
                    "tn": np.sum,
                    "fp": np.sum,
                    "fn": np.sum,
                    "tp": np.sum,
                    "time": np.average,
                }
            )
            .reset_index()
        )

        # Calculating MCC
        dd_clf["MCC"] = dd_clf.apply(
            lambda x: MCC(x["tn"], x["fp"], x["fn"], x["tp"]), axis=1
        )
        dd_post["MCC"] = dd_post.apply(
            lambda x: MCC(x["tn"], x["fp"], x["fn"], x["tp"]), axis=1
        )
        # Maximum MCC
        dd_clf_max = dd_clf.loc[dd_clf["MCC"] == dd_clf["MCC"].max(), :]
        dd_post_max = dd_post.loc[dd_post["MCC"] == dd_post["MCC"].max(), :]

        # Storing best hyperparameters
        threshold_classifier = dd_clf_max["threshold"].values[0] / 100
        threshold_postproc = dd_post_max["threshold"].values[0] / 100
        opening_value = dd_post_max["opening"].values[0]
        closing_value = dd_post_max["closing"].values[0]
        voting_width = dd_post_max["voting_width"].values[0]
        voting_depth = dd_post_max["voting_depth"].values[0]
        count = dd_post_max["count"].values[0] / 100

        # Mounting parameters dict
        param_dict = {
            "warping": alignment,
            "boosting": classifier,
            "threshold": threshold_postproc,
            "opening": opening_value if exp["opening"] == 1 else 1,
            "closing": closing_value if exp["closing"] == 1 else 1,
            "vw_depth": voting_depth if exp["vw_depth"] == 1 else 1,
            "vw_width": voting_width if exp["vw_width"] == 1 else 1,
            "count": count if exp["count"] == 1 else 0.50,
        }

        # Extracting silhouette and prediction
        sil_data = data["silhouette"]
        vid_data = data["prediction"]

        # List of videos in this folder
        videos = list(vid_data.keys())

        print(param_dict)
        for vv in videos:
            vid = vid_data[vv]
            sil = sil_data[vv]

            time_0 = time.time()

            # Threshold
            vid_clf = threshold(vid, param_dict["threshold"])

            # Opening
            vid = opening(vid_clf, param_dict["opening"])

            # Closing
            vid = closing(vid, param_dict["closing"])

            # Opening
            vid_out = voting_window(
                vid,
                (
                    param_dict["vw_depth"],
                    param_dict["vw_width"],
                    param_dict["vw_width"],
                ),
                param_dict["count"],
            )

            time_1 = time.time()

            # Calculating metrics
            mcc, dis, tn, fp, fn, tp = get_metrics(vid_out, sil)
            dict_results[vv] = {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "mcc": mcc,
                "dis": dis,
                "time": time_1 - time_0,
            }

    res_df = pd.DataFrame(dict_results).T
    res_df.loc["sum", :] = res_df.sum()
    res_df.loc["average", :] = res_df.iloc[:59, :].mean()
    res_df.loc["overall", "mcc"] = MCC(
        res_df.loc["sum", "tn"],
        res_df.loc["sum", "fp"],
        res_df.loc["sum", "fn"],
        res_df.loc["sum", "tp"],
    )
    res_df.loc["overall", "dis"] = DIS(
        res_df.loc["sum", "tn"],
        res_df.loc["sum", "fp"],
        res_df.loc["sum", "fn"],
        res_df.loc["sum", "tp"],
    )
    res_df.loc["median", :] = res_df.iloc[:59, :].median()
    res_df.loc["std", :] = res_df.iloc[:59, :].std()

    # Storing resulting DataFrame in dict
    exp_results[key] = res_df

# Saving results
pickle.dump(exp_results,
            open(os.path.join(output_dir, "ablation_results.pkl"), "wb"))
