import cv2
import numpy as np


def postproc(E, p):
    beta = p[1]
    omega = int(p[2])
    mu1 = int(p[3])
    mu2 = int(p[4])
    kappa = int(p[5])
    h, w, vs = E.shape
    B = E.copy()

    print('Thresholding... ')
    if omega > 1:
        for j in range(vs):
            f = E[:, :, j]
            g = cv2.blur(f, (omega, omega))
            B[:, :, j] = g

    B = 1.0 * (np.abs(B) > beta)

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mu1, mu1))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mu2, mu2))

    for j in range(vs):
        b = B[:, :, j]
        if mu1 > 0:
            b = cv2.morphologyEx(b, cv2.MORPH_OPEN, se1)
        if mu2 > 0:
            b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, se2)
        B[:, :, j] = b

    if kappa > 0:
        # kw = 2 * kappa + 1
        B_ = B.copy()
        for k in range(1 + kappa, vs - kappa):
            idx = list(range(k - kappa, k + kappa + 1))
            B[:, :, k] = np.round(np.mean(B_[:, :, idx], axis=2))

    t4 = cv2.getTickCount()
    return B, t4
