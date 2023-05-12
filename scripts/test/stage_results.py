import pickle
import os
import pandas as pd
import numpy as np
import warnings
import sys
import time

from src.utils import MCC, DIS, conf_mat
from src import DATA_DIR, RESULT_DIR
from src.utils import threshold, opening, closing, voting_window

warnings.filterwarnings("ignore")

save = True
out_dir = os.path.join(DATA_DIR, 'test_classifier', 'result_data')
result_file = os.path.join(RESULT_DIR, 'test_results', 'numerical_results',
                           'metrics_test.pkl')

res_dict = {
    'threshold_classifier': 0,
    'threshold_postproc': 0,
    'opening': 0,
    'closing': 0,
    'voting_window': 0
}
mcc_results = {k: res_dict.copy() for k in range(1, 60)}
dis_results = {k: res_dict.copy() for k in range(1, 60)}
time_results = {k: res_dict.copy() for k in range(1, 60)}
mcc_results.update({
    'tn_sum': res_dict.copy(),
    'fp_sum': res_dict.copy(),
    'fn_sum': res_dict.copy(),
    'tp_sum': res_dict.copy(),
    'average': res_dict.copy(),
    'std': res_dict.copy(),
    'overall': res_dict.copy()
})
dis_results.update({
    'tn_sum': res_dict.copy(),
    'fp_sum': res_dict.copy(),
    'fn_sum': res_dict.copy(),
    'tp_sum': res_dict.copy(),
    'average': res_dict.copy(),
    'std': res_dict.copy(),
    'overall': res_dict.copy()
})
time_results.update({'average': res_dict.copy(), 'std': res_dict.copy()})

if len(sys.argv) > 5:
    classifier = str(sys.argv[2])
    alignment = str(sys.argv[4])
else:
    classifier = 'LightGBM'
    alignment = 'warp'

output_dir = os.path.join(RESULT_DIR, 'test_results',
                          '{0}_{1}'.format(classifier, alignment))


def get_metrics(res, sil):

    # treating sil
    # sil[sil >= 0.8] = 1
    # sil[(sil >= 0.2) & (sil < 0.8)] = 2
    # sil[sil < 0.2] = 0

    # confusion matrix
    res_out = res.reshape(-1)
    sil_out = sil.reshape(-1).numpy()

    valid_pix = np.where(sil_out != 0.5)
    tn, fp, fn, tp = conf_mat(sil_out[valid_pix], res_out[valid_pix]).ravel()

    mcc = MCC(tn, fp, fn, tp)
    dis = DIS(tn, fp, fn, tp)

    return mcc, dis, tn, fp, fn, tp


for fold in range(1, 10):

    out_file = os.path.join(
        out_dir, '{0}_{1}_fold{2:02d}.pkl'.format(classifier, alignment, fold))
    data = pickle.load(open(out_file, 'rb'))

    # Loading results
    res_clf_dir = os.path.join(RESULT_DIR, classifier, alignment,
                               'classification')
    res_pst_dir = os.path.join(RESULT_DIR, classifier, alignment,
                               'postprocessing')

    clf_csv = os.path.join(res_clf_dir, 'results_fold{0:02d}.csv'.format(fold))
    post_csv = os.path.join(res_pst_dir,
                            'results_post_fold{0:02d}.csv'.format(fold))

    dd_clf = pd.read_csv(clf_csv, index_col=0)
    dd_post = pd.read_csv(post_csv, index_col=0)
    dd_clf.reset_index(drop=True, inplace=True)
    dd_post.reset_index(drop=True, inplace=True)

    # Grouping by trial
    dd_clf = dd_clf.groupby(
        ['fold_num', 'trial', 'num_leaves', 'min_child_samples',
         'threshold']).aggregate({
             'num_trees': np.average,
             'tn': np.sum,
             'fp': np.sum,
             'fn': np.sum,
             'tp': np.sum,
             'time': np.average
         }).reset_index()
    dd_post = dd_post.groupby([
        'fold', 'step', 'threshold', 'voting_depth', 'voting_width', 'opening',
        'closing', 'count'
    ]).aggregate({
        'tn': np.sum,
        'fp': np.sum,
        'fn': np.sum,
        'tp': np.sum,
        'time': np.average
    }).reset_index()

    # Calculating MCC
    dd_clf['MCC'] = dd_clf.apply(
        lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)
    dd_post['MCC'] = dd_post.apply(
        lambda x: MCC(x['tn'], x['fp'], x['fn'], x['tp']), axis=1)
    # Maximum MCC
    dd_clf_max = dd_clf.loc[dd_clf['MCC'] == dd_clf['MCC'].max(), :]
    dd_post_max = dd_post.loc[dd_post['MCC'] == dd_post['MCC'].max(), :]

    # Storing best hyperparameters
    threshold_classifier = dd_clf_max['threshold'].values[0] / 100
    threshold_postproc = dd_post_max['threshold'].values[0] / 100
    opening_value = dd_post_max['opening'].values[0]
    closing_value = dd_post_max['closing'].values[0]
    voting_width = dd_post_max['voting_width'].values[0]
    voting_depth = dd_post_max['voting_depth'].values[0]
    count = dd_post_max['count'].values[0] / 100

    sil_data = data['silhouette']
    vid_data = data['prediction']

    videos = list(vid_data.keys())

    for vv in videos:

        vid = vid_data[vv]
        sil = sil_data[vv]

        # Threshold classifier
        time_0 = time.time()
        vid_clf = threshold(vid, threshold_classifier)
        time_1 = time.time()
        mcc, dis, tn, fp, fn, tp = get_metrics(vid_clf, sil)
        mcc_results[vv]['threshold_classifier'] = mcc
        dis_results[vv]['threshold_classifier'] = dis
        time_results[vv]['threshold_classifier'] = time_1 - time_0
        mcc_results['tn_sum']['threshold_classifier'] += tn
        mcc_results['fp_sum']['threshold_classifier'] += fp
        mcc_results['fn_sum']['threshold_classifier'] += fn
        mcc_results['tp_sum']['threshold_classifier'] += tp
        dis_results['tn_sum']['threshold_classifier'] += tn
        dis_results['fp_sum']['threshold_classifier'] += fp
        dis_results['fn_sum']['threshold_classifier'] += fn
        dis_results['tp_sum']['threshold_classifier'] += tp

        # Threshold post-processing
        time_0 = time.time()
        vid_post = threshold(vid, threshold_postproc)
        time_1 = time.time()
        mcc, dis, tn, fp, fn, tp = get_metrics(vid_post, sil)
        mcc_results[vv]['threshold_postproc'] = mcc
        dis_results[vv]['threshold_postproc'] = dis
        time_results[vv]['threshold_postproc'] = time_1 - time_0
        mcc_results['tn_sum']['threshold_postproc'] += tn
        mcc_results['fp_sum']['threshold_postproc'] += fp
        mcc_results['fn_sum']['threshold_postproc'] += fn
        mcc_results['tp_sum']['threshold_postproc'] += tp
        dis_results['tn_sum']['threshold_postproc'] += tn
        dis_results['fp_sum']['threshold_postproc'] += fp
        dis_results['fn_sum']['threshold_postproc'] += fn
        dis_results['tp_sum']['threshold_postproc'] += tp

        # Opening
        time_0 = time.time()
        vid = opening(vid_post, opening_value)
        time_1 = time.time()
        mcc, dis, tn, fp, fn, tp = get_metrics(vid, sil)
        mcc_results[vv]['opening'] = mcc
        dis_results[vv]['opening'] = dis
        time_results[vv]['opening'] = time_1 - time_0
        mcc_results['tn_sum']['opening'] += tn
        mcc_results['fp_sum']['opening'] += fp
        mcc_results['fn_sum']['opening'] += fn
        mcc_results['tp_sum']['opening'] += tp
        dis_results['tn_sum']['opening'] += tn
        dis_results['fp_sum']['opening'] += fp
        dis_results['fn_sum']['opening'] += fn
        dis_results['tp_sum']['opening'] += tp

        # Closing
        time_0 = time.time()
        vid = closing(vid, closing_value)
        time_1 = time.time()
        mcc, dis, tn, fp, fn, tp = get_metrics(vid, sil)
        mcc_results[vv]['closing'] = mcc
        dis_results[vv]['closing'] = dis
        time_results[vv]['closing'] = time_1 - time_0
        mcc_results['tn_sum']['closing'] += tn
        mcc_results['fp_sum']['closing'] += fp
        mcc_results['fn_sum']['closing'] += fn
        mcc_results['tp_sum']['closing'] += tp
        dis_results['tn_sum']['closing'] += tn
        dis_results['fp_sum']['closing'] += fp
        dis_results['fn_sum']['closing'] += fn
        dis_results['tp_sum']['closing'] += tp

        # Voting window
        time_0 = time.time()
        vid_out = voting_window(vid,
                                (voting_depth, voting_width, voting_width),
                                count)
        time_1 = time.time()
        mcc, dis, tn, fp, fn, tp = get_metrics(vid_out, sil)
        mcc_results[vv]['voting_window'] = mcc
        dis_results[vv]['voting_window'] = dis
        time_results[vv]['voting_window'] = time_1 - time_0
        mcc_results['tn_sum']['voting_window'] += tn
        mcc_results['fp_sum']['voting_window'] += fp
        mcc_results['fn_sum']['voting_window'] += fn
        mcc_results['tp_sum']['voting_window'] += tp
        dis_results['tn_sum']['voting_window'] += tn
        dis_results['fp_sum']['voting_window'] += fp
        dis_results['fn_sum']['voting_window'] += fn
        dis_results['tp_sum']['voting_window'] += tp

        if save:
            subfile_path = os.path.join(output_dir, '{0:02d}.pkl'.format(vv))
            vid_out[vid_out < 0.5] = 0
            vid_out[vid_out >= 0.5] = 1
            vid_out = vid_out.astype(np.uint8)
            out_dict = {
                'video': vid_out,
                'video_num': vv,
                'fold': fold,
                'pre_subsampling': 1,
                'cut_frame': 5,
                'post_subsampling': 8
            }
            pickle.dump(out_dict, open(subfile_path, 'wb'))

# MCC results
mcc_results['average'] = {
    'threshold_classifier':
    np.mean([mcc_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.mean([mcc_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.mean([mcc_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.mean([mcc_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.mean([mcc_results[vv]['voting_window'] for vv in range(1, 60)])
}
mcc_results['std'] = {
    'threshold_classifier':
    np.std([mcc_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.std([mcc_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.std([mcc_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.std([mcc_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.std([mcc_results[vv]['voting_window'] for vv in range(1, 60)])
}
mcc_results['overall'] = {
    'threshold_classifier':
    MCC(mcc_results['tn_sum']['threshold_classifier'],
        mcc_results['fp_sum']['threshold_classifier'],
        mcc_results['fn_sum']['threshold_classifier'],
        mcc_results['tp_sum']['threshold_classifier']),
    'threshold_postproc':
    MCC(mcc_results['tn_sum']['threshold_postproc'],
        mcc_results['fp_sum']['threshold_postproc'],
        mcc_results['fn_sum']['threshold_postproc'],
        mcc_results['tp_sum']['threshold_postproc']),
    'opening':
    MCC(mcc_results['tn_sum']['opening'], mcc_results['fp_sum']['opening'],
        mcc_results['fn_sum']['opening'], mcc_results['tp_sum']['opening']),
    'closing':
    MCC(mcc_results['tn_sum']['closing'], mcc_results['fp_sum']['closing'],
        mcc_results['fn_sum']['closing'], mcc_results['tp_sum']['closing']),
    'voting_window':
    MCC(mcc_results['tn_sum']['voting_window'],
        mcc_results['fp_sum']['voting_window'],
        mcc_results['fn_sum']['voting_window'],
        mcc_results['tp_sum']['voting_window']),
}

# DIS results
dis_results['average'] = {
    'threshold_classifier':
    np.mean([dis_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.mean([dis_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.mean([dis_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.mean([dis_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.mean([dis_results[vv]['voting_window'] for vv in range(1, 60)])
}
dis_results['std'] = {
    'threshold_classifier':
    np.std([dis_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.std([dis_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.std([dis_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.std([dis_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.std([dis_results[vv]['voting_window'] for vv in range(1, 60)])
}
dis_results['overall'] = {
    'threshold_classifier':
    DIS(dis_results['tn_sum']['threshold_classifier'],
        dis_results['fp_sum']['threshold_classifier'],
        dis_results['fn_sum']['threshold_classifier'],
        dis_results['tp_sum']['threshold_classifier']),
    'threshold_postproc':
    DIS(dis_results['tn_sum']['threshold_postproc'],
        dis_results['fp_sum']['threshold_postproc'],
        dis_results['fn_sum']['threshold_postproc'],
        dis_results['tp_sum']['threshold_postproc']),
    'opening':
    DIS(dis_results['tn_sum']['opening'], dis_results['fp_sum']['opening'],
        dis_results['fn_sum']['opening'], dis_results['tp_sum']['opening']),
    'closing':
    DIS(dis_results['tn_sum']['closing'], dis_results['fp_sum']['closing'],
        dis_results['fn_sum']['closing'], dis_results['tp_sum']['closing']),
    'voting_window':
    DIS(dis_results['tn_sum']['voting_window'],
        dis_results['fp_sum']['voting_window'],
        dis_results['fn_sum']['voting_window'],
        dis_results['tp_sum']['voting_window']),
}

# time results
time_results['average'] = {
    'threshold_classifier':
    np.mean([time_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.mean([time_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.mean([time_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.mean([time_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.mean([time_results[vv]['voting_window'] for vv in range(1, 60)])
}
time_results['std'] = {
    'threshold_classifier':
    np.std([time_results[vv]['threshold_classifier'] for vv in range(1, 60)]),
    'threshold_postproc':
    np.std([time_results[vv]['threshold_postproc'] for vv in range(1, 60)]),
    'opening':
    np.std([time_results[vv]['opening'] for vv in range(1, 60)]),
    'closing':
    np.std([time_results[vv]['closing'] for vv in range(1, 60)]),
    'voting_window':
    np.std([time_results[vv]['voting_window'] for vv in range(1, 60)])
}

mcc_df = pd.DataFrame(mcc_results).T
dis_df = pd.DataFrame(dis_results).T
time_df = pd.DataFrame(time_results).T

# Saving
mcc_df.to_csv(result_file.replace('.pkl', '_mcc.csv'), mode='w', header=True)
dis_df.to_csv(result_file.replace('.pkl', '_dis.csv'), mode='w', header=True)
time_df.to_csv(result_file.replace('.pkl', '_time.csv'), mode='w', header=True)
