import numpy as np
from scipy.sparse.linalg import svds
from solver import ALst


def sprep(D, X, lambda_, gamma, tol, W=None, E=None):
    m, d = D.shape
    m, n = X.shape

    if W is None or E is None:
        W = np.zeros((d, n))
        E = np.zeros((m, n))

    if tol is None:
        tol = 1e-2

    # initialization
    # I = np.eye(n)
    # L-multiplier
    Y = X
    norm_two = svds(Y, k=1, return_singular_vectors=False)[0]
    norm_inf = np.max(np.abs(Y))
    dual_norm = max(norm_two, norm_inf)
    Y = -Y / dual_norm

    # mu, rho and eta
    mu = 1.25 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    eta1 = 3
    eta2 = 1.1 * np.linalg.norm(D, 2)**2

    # other parameters
    iter = 0
    converge = False
    maxIter = 9999
    # norm_D = np.linalg.norm(D, 'fro')
    norm_X = np.linalg.norm(X, 'fro')

    while not converge:
        iter += 1

        # --- update W ---
        L = X - E
        # was: W_temp = W - L'*(L*W - L + Y./mu)./eta2;
        W_temp = W - D.T @ (D @ W - L + Y / mu) / eta2
        W = ALst(W_temp, lambda_ / (mu * eta2))

        # --- update E ---
        # was: W_h = I - W;
        # was: E_temp = E + (L*W_h - Y./mu)*W_h'./eta1;
        E_temp = E - (D @ W - L + Y / mu) / eta1
        E = ALst(E_temp, gamma / (mu * eta1))

        # --- update Y and mu ---
        # Z = D*W - L;  % exp001
        Z = D @ W - (X - E)  # mod 4
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        # --- stop criterion ---
        T = np.linalg.norm(Z, 'fro') / norm_X
        norm_W = np.linalg.norm(W, 'fro')
        norm_E = np.linalg.norm(E, 'fro')
        print(f'iter:{iter:04d} T:{T:e} |W|:{norm_W:g} |E|:{norm_E:g}')
        converge = T < tol and norm_W > 0
        if iter >= maxIter:
            break

    return W, E, T, iter
