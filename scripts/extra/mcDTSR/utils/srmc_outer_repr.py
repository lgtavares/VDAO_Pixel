import numpy as np
import cv2
from scipy.linalg import norm, pinv
from .projective_matrix_to_parameters import projective_matrix_to_parameters
from .image_Jaco import image_Jaco

from skimage import transform
from scipy import signal

from .srmc_inner_repr import srmc_inner_repr
from .parameters_to_projective_matrix import parameters_to_projective_matrix


def vec(A):
    vecA = A.flatten()
    return vecA


def srmc_outer_repr(Vr, Vt, params, data, T0=None):
    # read input video dimensions
    h, w, n1 = Vr.shape
    n2 = Vt.shape[2]

    # image spatial derivatives: Ix Iy
    I0x = []
    I0y = []
    for idx in range(n2):
        I0 = Vt[:, :, idx]

        # image derivatives
        I0x.append(
            signal.convolve2d(I0,
                              np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) /
                              8,
                              mode='same',
                              boundary='symm'))
        I0y.append(
            signal.convolve2d(I0,
                              np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) /
                              8,
                              mode='same',
                              boundary='symm'))

    # outer loop setup and statistics (history)
    xi = []
    iterNum = data['numIterOuter']
    converged = False

    # get the initial input images in canonical frame
    ww, hh = params['windowSize'][1], params['windowSize'][0]
    tx, ty = params['windowPos'][1], params['windowPos'][0]

    Xr = np.zeros((ww * hh, n1))
    Xt = np.zeros((ww * hh, n2))

    # basic transform
    Tb = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # >>> new xi initialization <<<
    # is T0 given?
    if T0 is None:
        print('info: no T0 given, setting T0 = I')
        T0 = np.eye(3)
    # initialize xi transforms
    T_in = []
    for i in range(n2):
        T_in.append(T0.dot(Tb))
        xi.append(projective_matrix_to_parameters(params, T_in[i]))

    Tfm = transform.ProjectiveTransform(pinv(Tb.T))
    for idx in range(n1):
        Ir = np.reshape(transform.warp(Vr[:, :, idx],
                                       Tfm,
                                       order=3,
                                       preserve_range=True,
                                       output_shape=(hh, ww)),
                        (hh * ww, 1),
                        order='F')
        Xr[:, idx] = Ir[:, 0] / np.linalg.norm(Ir)  # normalize

    # remaining code here
    # ...
    relerr = np.inf
    while not converged:
        iterNum += 1

        J = {}  # cell(1,n2)
        print('(outer) #{}'.format(iterNum))

        print('Warping images...')
        for idx in range(n2):
            # Transformed image and derivatives with respect to
            # tform parameters
            Tfm = pinv(T_in[idx]).T

            # Compute Jacobian
            Id = cv2.warpPerspective(Vt[:, :, idx],
                                     Tfm, (ww, hh),
                                     flags=cv2.INTER_CUBIC)
            Iu = cv2.warpPerspective(I0x[idx],
                                     Tfm, (ww, hh),
                                     flags=cv2.INTER_CUBIC)
            Iv = cv2.warpPerspective(I0y[idx],
                                     Tfm, (ww, hh),
                                     flags=cv2.INTER_CUBIC)

            y = Id.flatten(order='F')
            y_norm = np.linalg.norm(y)

            Iu = Iu.flatten(order='F') / y_norm -\
                (y.T.dot(Iu.flatten(order='F')) / (y_norm**3)) * y
            Iv = Iv.flatten(order='F') / y_norm -\
                (y.T.dot(Iv.flatten(order='F')) / (y_norm**3)) * y

            y = y / y_norm

            # Warped Xt % D = [D y] ;
            Xt[:, idx] = y

            # Transformation matrix to parameters
            xi[idx] = projective_matrix_to_parameters(params, T_in[idx])

            # Compute Jacobian
            J[idx] = image_Jaco(Iu, Iv, params, xi[idx])

        # lambda = params.lambdac/sqrt(size(Xt,1)) ;
        lambda_ = params['lambda']

        # RASL inner loop
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------
        # using QR to orthogonalize the Jacobian matrix
        Q = []
        R = []
        for idx in range(n2):
            Q_idx, R_idx = np.linalg.qr(J[idx], mode='reduced')
            Q.append(Q_idx)
            R.append(R_idx)

        W, E, delta_xi, numIterInnerEach = srmc_inner_repr(Xr, Xt, Q, params)

        for idx in range(n2):
            delta_xi[idx] = np.linalg.inv(R[idx]) @ delta_xi[idx]
        # -----------------------------------------------------------------
        # -----------------------------------------------------------------

        # curObj = norm(svd(A),1) + lambda*norm(E(:),1) ;
        W_norm1 = norm(W.ravel(), 1)
        E_norm1 = norm(E.ravel(), 1)
        # W_norm1 = norm(W,1)
        # E_norm1 = norm(E,1)
        curObj = W_norm1 + lambda_ * E_norm1
        diffObj = abs(data.prevObj - curObj)
        relerr = diffObj / curObj
        print('   prev. obj. function: {}'.format(data.prevObj))
        print('   curr. obj. function: {}'.format(curObj))
        print('   difference: {}  ({}%)'.format(diffObj, 100 * relerr))
        print('   relative error: {}'.format(relerr))

        # step in parameters
        dt = [delta_xi[i][0] for i in range(n2)]
        dt_norm1 = np.linalg.norm(np.array(dt).reshape(-1), 1)
        for i in range(n2):
            xi[i] = xi[i] + delta_xi[i][0]
            T_in = parameters_to_projective_matrix(params, xi[i])
            T_in = np.array(T_in)

        if ((relerr < params.stoppingDelta) or (iterNum >= params.maxIter)):
            converged = 1
            if (iterNum >= params['maxIter']):
                print('(outer loop) maximum iterations reached')
            else:
                print('(outer loop) converged')

        # step out transformations
        for i in range(n2):
            T_out = parameters_to_projective_matrix(params, xi[i])
            T_out = np.array(T_out)
            xi[i] = projective_matrix_to_parameters(
                params, np.dot(T_out, np.linalg.inv(Tb)))

        # update statistics
        data['numIterOuter'] = iterNum
        data['numIterInner'][iterNum] = numIterInnerEach
        data['W'][iterNum] = W
        data['xi'][iterNum] = xi
        data['W_1'][iterNum] = W_norm1
        data['E_1'][iterNum] = E_norm1
        data['dt_1'][iterNum] = dt_norm1
        data['time'][iterNum] = 0
        data['rhos'][iterNum] = params.rho
        data['rerr'][iterNum] = relerr
        data['prevObj'] = curObj

        # time spent
        print('(outer loop) total iterations: {}'.format(iterNum))

    return Xr, Xt, W, E, xi, data
