from solver import ALst
import numpy as np


def kspdp4(X, lambda_=1e-3, gamma=0.1 * 1e-3, tol=1e-2, W=None, E=None):
    """
    k-subspace decomposition algorithm for matrix X
    with regularization parameters lambda_ and gamma, tolerance tol,
    and initial coefficient and sparse matrices W and E (optional).
    Returns W, E, T (residual error), and iter (number of iterations).
    """
    d, n = X.shape

    if W is None:
        W = np.zeros((n, n))
    if E is None:
        E = np.zeros((d, n))

    # initialization
    Id = np.eye(n)
    Y = X.copy()
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y, np.inf)
    dual_norm = max(norm_two, norm_inf)
    Y = -Y / dual_norm
    mu = 1.25 / norm_two
    mu_bar = mu * 1e10
    rho = 1.5
    eta1 = 3
    eta2 = 1.1 * norm_two**2
    iter = 0
    converge = False
    maxIter = 9999
    norm_X = np.linalg.norm(X, 'fro')

    # iterative algorithm
    while not converge:
        iter += 1

        # update W
        L = X - E
        W_temp = W - L.T @ (L @ W - L + Y / mu) / eta2
        W = ALst(W_temp, lambda_ / (mu * eta2))
        W = W - np.diag(np.diag(W))

        # update E
        W_h = Id - W
        E_temp = E + (L @ W_h - Y / mu) @ W_h.T / eta1
        E = ALst(E_temp, gamma / (mu * eta1))

        # update Y and mu
        Z = L @ W - L
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        # stop criterion
        T = np.linalg.norm(Z, 'fro') / norm_X
        line = f"iter:{iter:04d} T:{T:e} "
        line += f"|W|:{np.linalg.norm(W, 'fro'):g} "
        line += f"|E|:{np.linalg.norm(E, 'fro'):g} "
        print(line)

        converge = T < tol

        if iter >= maxIter:
            break

    return W, E, T, iter
