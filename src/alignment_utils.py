
import torch
import numpy as np
import math
import cv2
import PIL.Image as Image
from src.config import transformations_half
from skimage.measure import ransac
from skimage.registration import optical_flow_tvl1
from skimage.transform import EuclideanTransform
from skimage import transform as tf


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

        diff = diff*mask
        rmse = math.sqrt((diff**2).sum() / total_pixels)

    rmse = math.sqrt(np.mean(np.square(diff)))
    if diff.dtype == np.uint8:
        abs_diff_img = abs(diff)
    else:
        abs_diff_img = (255 * abs(diff)).astype(np.uint8)
    return {'rmse': rmse, 'abs_diff_img': abs_diff_img}


def disconsider_borders(image, percentage=.05):
    if len(image.shape) == 3:
        h, w, _ = image.shape
    elif len(image.shape) == 2:
        h, w = image.shape
    h_remove = int(h * percentage)
    w_remove = int(w * percentage)
    if len(image.shape) == 3:
        return image[h_remove:h - h_remove, w_remove:w - w_remove, :]
    elif len(image.shape) == 2:
        return image[h_remove:h - h_remove, w_remove:w - w_remove]


def load_ref_frames(cap, beg, end):
    n_frames = end-beg+1
    z = np.zeros((n_frames, 360, 640))
    for i, frm in enumerate(range(beg, end+1)):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
        ref_frame = cap.read()[1]
        ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        z[i] = transformations_half(Image.fromarray(ref_frame).convert('L'))
    return z


def apply_optical_flow(ref_image,
                       moving_image,
                       sil_image,
                       bounding_box=None,
                       min_samples=10,
                       apply_ransac=True,
                       ransac_res_only=False,
                       transformations=None,
                       prefilter=False,
                       get_optical_vectors_ransac=False,
                       res_threshold=0.1,
                       border_percentage_removal=0.05):

    if isinstance(ref_image, torch.Tensor):
        ref_image = ref_image.numpy()
    if isinstance(moving_image, torch.Tensor):
        moving_image = moving_image.numpy()
    if isinstance(sil_image, torch.Tensor):
        sil_image = sil_image.numpy()
    if ref_image.dtype != np.uint8:
        ref_image = (255 * ref_image).astype(np.uint8)
    if moving_image.dtype != np.uint8:
        moving_image = (255 * moving_image).astype(np.uint8)
    if sil_image.dtype != np.uint8:
        sil_image = (255 * sil_image).astype(np.uint8)

    if transformations is not None:
        ref_image = transformations(Image.fromarray(ref_image))
        ref_image = (255 * ref_image.squeeze().permute(1, 2, 0).numpy())
        ref_image = ref_image.astype(np.uint8)
        moving_image = transformations(Image.fromarray(moving_image)).squeeze()
        moving_image = (255 * moving_image.permute(1, 2, 0).numpy())
        moving_image = moving_image.astype(np.uint8)

    if isinstance(ref_image, Image.Image):
        ref_image = np.array(ref_image)
    if isinstance(moving_image, Image.Image):
        moving_image = np.array(moving_image)

    # Optical flow is applied into grayscale images
    ref_image_gs = ref_image
    moving_image_gs = moving_image

    nr, nc = ref_image_gs.shape

    # deslocamento vertical (v) e horizontal (u) considera que a ref_frames
    # é a deslocada
    v, u = optical_flow_tvl1(moving_image_gs, ref_image_gs, tightness=0.1,
                             num_warp=20, tol=1e-7, prefilter=prefilter)

    row_coords, col_coords = np.meshgrid(np.arange(nr),
                                         np.arange(nc),
                                         indexing='ij')

    if apply_ransac:

        src = np.array((row_coords.flatten(), col_coords.flatten())).T
        dst = np.array(((row_coords + u).flatten(),
                        (col_coords + v).flatten())).T
        # Fita modelo eliminando outliers com ransac
        error = np.sqrt((src - dst)**2).sum(axis=1)
        t = np.sqrt(5.99 * error.std())
        # Aplica ransac para retirar outliers
        model_ransac, inliers = ransac(
            (dst, src),
            EuclideanTransform,
            # min_samples=int((1 - 0.2) * (nc * nr)),  # Testar outros valores
            min_samples=min_samples,
            residual_threshold=t*res_threshold,  # Testar outros valores
            max_trials=50)

        # Gera a imagem com vetores com a aplicação do ransac
        img_vectors_ransac = None

        # Gera imagem retificada em cima da referencia em grayscale
        ref_ret_ransac_gs = tf.warp(ref_image_gs, model_ransac.inverse)

        # Calcula RMS entre o frame alvo grayscale e o de referência
        # transformado e em grayscale e desconsiderando
        rms_ransac = calculate_SRMSE(
            moving_image_gs,
            ref_ret_ransac_gs,
            border_percentage_removal=border_percentage_removal)

        abs_diff_noborders = rms_ransac['abs_diff_img']
        rms_ransac_full_frame = rms_ransac['rmse']
        # Tentativa: Dividindo pelo std
        # _ref_image_no_border = disconsider_borders(ref_image_gs,
        # border_percentage_removal)
        # rms_ransac = calculate_SRMS(abs_diff_noborders) /
        # _ref_image_no_border.std()
        # Se foi passado algum bounding box, calcula o RMS
        # desconsiderando a area do bounding box

        repositioned_bb = None
        rms_ransac_outside_bb_only = rms_ransac['rmse']
        if bounding_box is not None:

            rms_ransac_outside_bb_only = calculate_SRMSE(
                moving_image_gs,
                ref_ret_ransac_gs,
                disconsider_region=sil_image[0],
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
