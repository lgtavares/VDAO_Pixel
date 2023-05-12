import numpy as np


def dlt(p, q):
    A = np.zeros((0, 9))
    for i in range(p.shape[1]):
        A = np.vstack((A,
                       np.hstack((np.zeros((1, 3)),
                                  -q[2, i] * p[:, i][:, np.newaxis].T,
                                  q[1, i] * p[:, i][:, np.newaxis].T)),
                       np.hstack((q[2, i] * p[:, i][:, np.newaxis].T,
                                  np.zeros((1, 3)),
                                  -q[0, i] * p[:, i][:, np.newaxis].T))))
    U, S, V = np.linalg.svd(A)
    x = V[-1, :]
    H = np.reshape(x, (3, 3)).T
    H /= H[2, 2]
    return H


def h2i(p):
    m, n = p.shape
    q = np.zeros((m - 1, n))
    for i in range(n):
        q[:, i] = p[:m - 1, i] / p[-1, i]
    return q


def i2h(p):
    m, n = p.shape
    q = np.ones((m + 1, n))
    for i in range(n):
        q[:m, i] = p[:, i]
    return q


def projective_matrix_to_parameters(params, T):
    if params['transformType'] == 'TRANSLATION':
        xi = T[:2, 2]
    elif params['transformType'] == 'EUCLIDEAN':
        xi = np.zeros((3, 1))
        theta = np.arccos(T[0, 0])
        if T[1, 0] < 0:
            theta = -theta
        xi[0] = theta
        xi[1] = T[0, 2]
        xi[2] = T[1, 2]
    elif params['transformType'] == 'SIMILARITY':
        xi = np.zeros((4, 1))
        sI = T[:2, :2].T @ T[:2, :2]
        xi[0] = np.sqrt(sI[0])
        theta = np.arccos(T[0, 0] / xi[0])
        if T[1, 0] < 0:
            theta = -theta
        xi[1] = theta
        xi[2] = T[0, 2]
        xi[3] = T[1, 2]
    elif params['transformType'] == 'AFFINE':
        xi = np.zeros((6, 1))
        xi[:3] = T[0]
        xi[3:] = T[1]
    elif params['transformType'] == 'HOMOGRAPHY':
        xi = np.zeros((8, 1))
        xi[:3] = T[0]
        xi[3:6] = T[1]
        xi[6:8] = T[2, :2]
    elif params['transformType'] == '4P-HOMOGRAPHY':
        xi = np.zeros((8, 1))
        ww = params['windowSize'][1]
        hh = params['windowSize'][0]
        T0 = np.array([[1, 0, params['windowPos'][1]],
                       [0, 1, params['windowPos'][0]],
                       [0, 0, 1]])
        p = np.array([[1, ww, 1, ww],
                      [1, 1, hh, hh],
                      [1, 1, 1, 1]])
        xi[:8] = np.reshape(h2i(T @ p) - h2i(T0 @ p), (8, 1))
    else:
        raise ValueError('Unrecognized transformation')

    return xi
