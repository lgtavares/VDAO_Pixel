import math
import os
from PIL import Image
import cv2
from skimage.measure import ransac
from skimage.registration import optical_flow_tvl1
from skimage.transform import EuclideanTransform
from skimage import transform as tf

import torch
import numpy as np
import json
from scipy import signal
from pathlib import Path


def get_fold(filename, fold_number):
    assert fold_number >= 1 and fold_number <= 72, \
        'Fold number should be greater than 1 and less than 72.'
    if os.path.isfile(filename):
        with open(filename, "r") as read_file:
            all_folds = json.load(read_file)
            return all_folds[str(fold_number)]


def init_worker_random(seed=123):

    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_morphology(img_in, open_size=3, close_size=3):

    # morphology kernels
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                            (open_size, open_size))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                             (close_size, close_size))

    # apply morphology
    img_morph = (255 * img_in.reshape(-1, 90, 160)).astype(np.uint8)
    img_morph = [
        cv2.threshold(img, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        for img in img_morph
    ]
    img_morph = [
        cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open) for img in img_morph
    ]
    img_morph = [
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)
        for img in img_morph
    ]

    return np.stack(img_morph) // 255


def DIS(tn, fp, fn, tp):

    if (tp + fn) == 0:
        tp_rate = 0
    else:
        tp_rate = tp / (tp + fn)

    if (fp + tn) == 0:
        fp_rate = 0
    else:
        fp_rate = fp / (fp + tn)

    return np.sqrt((1 - tp_rate)**2 + fp_rate**2)


def MCC(tn, fp, fn, tp):

    den = np.sqrt(tp + fp) * np.sqrt(tp + fn) * np.sqrt(tn + fp) * np.sqrt(tn +
                                                                           fn)
    num = (tp * tn) - (fp * fn)

    if den == 0:
        return 0
    else:
        return num / den


def threshold(vid, value):
    return (vid >= value).numpy().astype('uint8')


def opening(vid, value):

    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (value, value))
    return np.array([
        cv2.morphologyEx(vid[f, :, :], cv2.MORPH_OPEN, kernel_opening)
        for f in range(vid.shape[0])
    ])


def closing(vid, value):

    kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (value, value))
    return np.array([
        cv2.morphologyEx(vid[f, :, :], cv2.MORPH_CLOSE, kernel_closing)
        for f in range(vid.shape[0])
    ])


def erosion(vid, value):

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (value, value))
    return np.array([
        cv2.morphologyEx(vid[f, :, :], cv2.MORPH_ERODE, kernel_erosion)
        for f in range(vid.shape[0])
    ])


def dilation(vid, value):

    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (value, value))
    return np.array([
        cv2.morphologyEx(vid[f, :, :], cv2.MORPH_DILATE, kernel_dilation)
        for f in range(vid.shape[0])
    ])


def voting_window(vid, shape, limit):

    vw = np.ones((shape[0], shape[1], shape[2]))
    out_vid = signal.convolve(vid.copy(), vw, mode='same')
    return (out_vid > limit * (np.size(vw))).astype('uint8')


def split_sequence(a):
    return [list(k) for k in np.split(a, np.where(np.diff(a) != 1)[0] + 1)]


def apply_optical_flow(ref_image,
                       tar_image,
                       bounding_box=None,
                       nvec=20,
                       apply_ransac=True,
                       ransac_res_only=False,
                       transformations=None,
                       transform=EuclideanTransform,
                       border_percentage_removal=0.05):

    if isinstance(ref_image, torch.Tensor):
        ref_image = ref_image.numpy()
    if isinstance(tar_image, torch.Tensor):
        tar_image = tar_image.numpy()

    if ref_image.dtype != np.uint8:
        ref_image = (255 * ref_image).astype(np.uint8)
    if tar_image.dtype != np.uint8:
        tar_image = (255 * tar_image).astype(np.uint8)

    if transformations is not None:
        ref_image = transformations(
            Image.fromarray(ref_image)).squeeze().permute(1, 2, 0)
        ref_image = (255 * ref_image.numpy()).astype(np.uint8)
        tar_image = transformations(
            Image.fromarray(tar_image)).squeeze().permute(1, 2, 0)
        tar_image = (255 * tar_image.numpy()).astype(np.uint8)

    if isinstance(ref_image, Image.Image):
        ref_image = np.array(ref_image)
    if isinstance(tar_image, Image.Image):
        tar_image = np.array(tar_image)

    # Optical flow is applied into grayscale images
    ref_image_gs = ref_image[0]
    tar_image_gs = tar_image[0]

    nr, nc = ref_image_gs.shape

    # deslocamento vertical (v) e horizontal (u) considera que
    # a ref_frames é a deslocada
    v, u = optical_flow_tvl1(tar_image_gs,
                             ref_image_gs,
                             tightness=0.1,
                             num_warp=20,
                             tol=1e-7)

    row_coords, col_coords = np.meshgrid(np.arange(nr),
                                         np.arange(nc),
                                         indexing='ij')

    if apply_ransac:

        src = np.array((row_coords.flatten(), col_coords.flatten())).T
        dst = np.array(
            ((row_coords + u).flatten(), (col_coords + v).flatten())).T

        # Fita modelo eliminando outliers com ransac
        error = np.sqrt((src - dst)**2).sum(axis=1)
        t = np.sqrt(5.99 * error.std())

        if t < error.min():
            t = 10 * (error.mean() - error.min())

        # Aplica ransac para retirar outliers
        model_ransac, inliers = ransac(
            (dst, src),
            transform,
            min_samples=10,
            residual_threshold=t / 10,  # Testar outros valores
            max_trials=100)

        # Gera a imagem com vetores com a aplicação do ransac
        img_vectors_ransac = None

        # Gera imagem retificada em cima da referencia em grayscale
        ref_ret_ransac_gs = tf.warp(ref_image_gs, model_ransac.inverse)

        # Calcula RMS entre o frame alvo grayscale e o de referência
        # transformado e em grayscale e desconsiderando
        rms_ransac = calculate_SRMSE(
            tar_image_gs,
            ref_ret_ransac_gs,
            border_percentage_removal=border_percentage_removal)

        abs_diff_noborders = rms_ransac['abs_diff_img']
        rms_ransac_full_frame = rms_ransac['rmse']
        # Tentativa: Dividindo pelo std
        # _ref_image_no_border = disconsider_borders(ref_image_gs,
        #  border_percentage_removal)
        # rms_ransac = calculate_SRMS(abs_diff_noborders) /
        #  _ref_image_no_border.std()
        # Se foi passado algum bounding box, calcula o RMS
        #  desconsiderando a area do bounding box

        repositioned_bb = None
        rms_ransac_outside_bb_only = rms_ransac['rmse']
        if bounding_box is not None:

            rms_ransac_outside_bb_only = calculate_SRMSE(
                tar_image_gs,
                ref_ret_ransac_gs,
                disconsider_region=None,
                border_percentage_removal=border_percentage_removal)
            rms_ransac_outside_bb_only = rms_ransac_outside_bb_only['rmse']
    else:
        img_vectors_ransac, model_ransac, inliers, rms_ransac,
        rms_ransac_outside_bb_only, abs_diff_noborders, ref_ret_ransac_gs,
        repositioned_bb = None, None, None, None, None, None, None, None

    if ref_ret_ransac_gs is not None:
        ref_ret_ransac_gs = (255 * ref_ret_ransac_gs).astype(np.uint8)
    if ransac_res_only:
        return {
            'img_ret_ransac_grayscale': ref_ret_ransac_gs,
            'img_ret_ransac_abs_diff_noborders': abs_diff_noborders,
            'model_ransac': model_ransac,
            'rms_ransac': rms_ransac_full_frame,
            'rms_ransac_outside_bb_only': rms_ransac_outside_bb_only,
            'repositioned_bb': repositioned_bb,
            'inliers_ransac': inliers,
            'img_optical_vectors_ransac': img_vectors_ransac,
            'optical_flow_components': [v, u],
        }


def calculate_SRMSE(img_1,
                    img_2,
                    disconsider_region=None,
                    transf_to_grayscale=True,
                    border_percentage_removal=None):

    diff = img_1 - img_2

    if isinstance(diff, torch.Tensor):
        diff = diff.numpy()
    total_pixels = diff.size

    if disconsider_region is not None:

        mask = (disconsider_region < 0.1).astype(np.uint8)
        total_pixels = mask.sum()

        diff = diff * mask
        rmse = math.sqrt((diff**2).sum() / total_pixels)

    rmse = math.sqrt(np.mean(np.square(diff)))
    if diff.dtype == np.uint8:
        abs_diff_img = abs(diff)
    else:
        abs_diff_img = (255 * abs(diff)).astype(np.uint8)
    return {'rmse': rmse, 'abs_diff_img': abs_diff_img}


def conf_mat(x, y):
    totaltrue = np.sum(x)
    return conf_mat_opt(x.astype(int), y.astype(int), totaltrue,
                        len(x) - totaltrue)


def conf_mat_opt(x, y, totaltrue, totalfalse):
    truepos, totalpos = np.sum(x & y), np.sum(y)
    falsepos = totalpos - truepos
    return np.array([
        [totalfalse - falsepos, falsepos],  # true negatives, false positives
        [totaltrue - truepos, truepos]
    ])  # false negatives, true positives


def create_dir(path):
    """Check if a directory exists, and if not, create it."""

    Path(path).mkdir(parents=True, exist_ok=True)
