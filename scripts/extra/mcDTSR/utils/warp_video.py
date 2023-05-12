import numpy as np
from skimage.transform import warp, ProjectiveTransform
from parameters_to_projective_matrix import parameters_to_projective_matrix


def warp_video(V_in, params, xi, hh, ww, itype='bicubic', flip=True):
    if flip:
        Tfm = ProjectiveTransform(
                matrix=parameters_to_projective_matrix(params, xi)).inverse
    else:
        Tfm = ProjectiveTransform(
                matrix=parameters_to_projective_matrix(params, xi))
    V_out = np.zeros((hh, ww, V_in.shape[2]))
    for k in range(V_in.shape[2]):
        V_out[:, :, k] = warp(V_in[:, :, k], Tfm, output_shape=(hh, ww),
                              order=itype)
    return V_out
