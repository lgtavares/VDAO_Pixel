import numpy as np


def ALst(E, beta):
    E_hat = np.sign(E) * np.maximum(np.abs(E) - beta, 0)
    return E_hat
