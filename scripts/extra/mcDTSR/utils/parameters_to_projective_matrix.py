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


def parameters_to_projective_matrix(params, xi):
    T = np.eye(3)
    if params['transformType'] == 'TRANSLATION':
        T[0, 2] = xi[0]
        T[1, 2] = xi[1]
    elif params['transformType'] == 'EUCLIDEAN':
        R = np.array([[np.cos(xi[0]), -np.sin(xi[0])],
                      [np.sin(xi[0]), np.cos(xi[0])]])
        T[:2, :2] = R
        T[0, 2] = xi[1]
        T[1, 2] = xi[2]
    elif params['transformType'] == 'SIMILARITY':
        R = np.array([[np.cos(xi[1]), -np.sin(xi[1])],
                      [np.sin(xi[1]), np.cos(xi[1])]])
        T[:2, :2] = xi[0] * R
        T[0, 2] = xi[2]
        T[1, 2] = xi[3]
    elif params['transformType'] == 'AFFINE':
        T[:2, :] = np.array([[xi[0], xi[1], xi[2]], [xi[3], xi[4], xi[5]]])
    elif params['transformType'] == 'HOMOGRAPHY':
        T = np.array([[xi[0], xi[1], xi[2]], [xi[3], xi[4], xi[5]],
                      [xi[6], xi[7], 1]])
    elif params['transformType'] == '4P-HOMOGRAPHY':
        ww, hh = params['windowSize'][1], params['windowSize'][0]
        T0 = np.array([[1, 0, params['windowPos'][1]],
                       [0, 1, params['windowPos'][0]], [0, 0, 1]])
        p = np.array([[1, ww, 1, ww], [1, 1, hh, hh], [1, 1, 1, 1]])
        q = i2h(np.reshape(xi, (2, 4)) + h2i(T0 @ p))
        T = dlt(p, q)
    else:
        raise ValueError('Unrecognized transformation')
    return T
