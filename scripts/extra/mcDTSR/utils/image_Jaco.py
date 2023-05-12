import numpy as np
from .parameters_to_projective_matrix import parameters_to_projective_matrix


def dlt(p, q):
    A = np.zeros((0, 9))
    for i in range(p.shape[1]):
        A = np.vstack(
            (A,
             np.hstack((np.zeros((1, 3)), -q[2, i] * p[:, i][:, np.newaxis].T,
                        q[1, i] * p[:, i][:, np.newaxis].T)),
             np.hstack((q[2, i] * p[:, i][:, np.newaxis].T, np.zeros(
                 (1, 3)), -q[0, i] * p[:, i][:, np.newaxis].T))))
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


def vec(A):
    vecA = A.flatten()
    return vecA


def image_Jaco(Iu, Iv, params, xi):
    ww, hh = params['windowSize'][1], params['windowSize'][0]
    u = vec(np.tile(np.arange(1, ww + 1), (hh, 1)))
    v = vec(np.tile(np.arange(1, hh + 1).reshape((-1, 1)), (1, ww)))

    if params['transformType'] == 'TRANSLATION':
        J = np.column_stack((Iu, Iv))
    elif params['transformType'] == 'EUCLIDEAN':
        J = np.column_stack(
            (Iu * (-np.sin(xi[0]) * u - np.cos(xi[0]) * v) + Iv *
             (np.cos(xi[0]) * u - np.sin(xi[0]) * v), Iu, Iv))
    elif params['transformType'] == 'SIMILARITY':
        J = np.column_stack(
            (Iu * (np.cos(xi[1]) * u - np.sin(xi[1]) * v) + Iv *
             (np.sin(xi[1]) * u + np.cos(xi[1]) * v), Iu *
             (-xi[0] * np.sin(xi[1]) * u - xi[0] * np.cos(xi[1]) * v) + Iv *
             (xi[0] * np.cos(xi[1]) * u - xi[0] * np.sin(xi[1]) * v), Iu, Iv))
    elif params['transformType'] == 'AFFINE':
        J = np.column_stack((Iu * u, Iu * v, Iu, Iv * u, Iv * v, Iv))
    elif params['transformType'] == 'HOMOGRAPHY':
        T = parameters_to_projective_matrix(params, xi)
        X = T[0, 0] * u + T[0, 1] * v + T[0, 2]
        Y = T[1, 0] * u + T[1, 1] * v + T[1, 2]
        Z = T[2, 0] * u + T[2, 1] * v + 1
        J = np.column_stack(
            (Iu * u / Z, Iu * v / Z, Iu / Z, Iv * u / Z, Iv * v / Z, Iv / Z,
             (-Iu * X * u / (Z**2) - Iv * Y * u / (Z**2)),
             (-Iu * X * v / (Z**2) - Iv * Y * v / (Z**2))))
    elif params['transformType'] == '4P-HOMOGRAPHY':
        T = parameters_to_projective_matrix(params, xi)
        X = T[0, 0] * u + T[0, 1] * v + T[0, 2]
        Y = T[1, 0] * u + T[1, 1] * v + T[1, 2]
        Z = T[2, 0] * u + T[2, 1] * v + 1
        J = np.vstack([
            Iu * u / Z, Iu * v / Z, Iu / Z,
            Iv * u / Z, Iv * v / Z, Iv / Z,
            (-Iu * X * u / (Z ** 2) - Iv * Y * u / (Z ** 2)),
            (-Iu * X * v / (Z ** 2) - Iv * Y * v / (Z ** 2))
        ]).T

        p = np.array([[1, ww, 1, ww], [1, 1, hh, hh], [1, 1, 1, 1]])
        q = h2i(np.dot(T, p))
        dH = np.zeros((8, 8))
        for i in range(8):
            dq = np.zeros((2, 4))
            dq[i // 4, i % 4] = 1
            epsilon = 1e-1
            old_dTi = np.inf * np.ones((3, 3))
            while True:
                T2 = dlt(p, i2h(q + epsilon * dq))
                dT_i = (T2 - T) / epsilon
                if np.linalg.norm(old_dTi - dT_i, 'fro') < 1e-6:
                    break
                old_dTi = dT_i
                epsilon /= 2
            dH[:, i] = np.array([dT_i[0, 0:3], dT_i[1, 0:3],
                                 dT_i[2, 0:2]]).ravel()

        J = np.dot(J, dH)
        
    return J