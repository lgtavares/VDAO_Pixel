import numpy as np
from numpy.linalg import norm


def srmc_inner_repr(Xr, Xt, Q, params):
    m, n1 = Xr.shape
    n2 = Xt.shape[1]

    # initialize
    Y = Xt.copy()
    norm_two = norm(Y, 2)
    norm_inf = norm(Y.ravel(), np.inf) / params["lambda"]
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm
    obj_v = np.dot(Xt.ravel(), Y.ravel())

    W_k = np.zeros((n1, n2))
    E_k = np.zeros((m, n2))

    Qdt_k = np.zeros((m, n2))
    dt_k = [np.zeros((Q[i].shape[1], 1)) for i in range(n2)]

    mu = 1.25 / norm(Xt)
    mu_max = 1e12
    rho = params["rho"]
    eta_W = 1.001 * norm(Xr, 2) ** 2
    eta_E = 1.001
    eta_dt = 1.001

    Xt_norm = norm(Xt, 'fro')

    iter = 0
    converged = False
    while not converged:
        iter += 1

        # srmc - mode 1
        temp_T = W_k - np.dot(Xr.T,
                              Xr.dot(W_k) + E_k - Xt - Qdt_k + Y / mu) / eta_W
        W_k = np.sign(temp_T) * np.maximum(abs(temp_T) - 1 / mu / eta_W, 0)

        # srmc - mode 1
        temp_T = E_k - (Xr.dot(W_k) + E_k - Xt - Qdt_k + Y / mu) / eta_E
        E_k = np.sign(temp_T) * np.maximum(abs(temp_T) -
                                           params["lambda"] / mu / eta_E, 0)

        # srmc - mode 1
        temp_T = Xr.dot(W_k) + E_k - Xt - Qdt_k + Y / mu
        for i in range(n2):
            dt_k[i] += np.dot(Q[i].T, temp_T[:, i]) / eta_dt
            Qdt_k[:, i] = Q[i].dot(dt_k[i])

        Z = Xr.dot(W_k) + E_k - Xt - Qdt_k
        Y += mu * Z
        mu = min(mu * rho, mu_max)
        obj_v = np.dot(Z.ravel(), Y.ravel())
        stoppingCriterion = norm(Z, 'fro') / Xt_norm
        dt = np.concatenate(dt_k, axis=0)

        if iter % params["DISPLAY_EVERY"] == 0:
            line = "#%d  ||W||_1 %.3f  " % (iter, norm(W_k.ravel(), 1))
            line += "||E||_1 %.3f  " % (norm(E_k.ravel(), 1))
            line += "||dt||_1 %.3f  " % (norm(dt.ravel(), 1))
            line += "obj: %.3f[%d]  tol: %.3f" % (obj_v, mu == mu_max,
                                                  stoppingCriterion)
            print(line)

    return W_k, E_k, dt_k, iter
