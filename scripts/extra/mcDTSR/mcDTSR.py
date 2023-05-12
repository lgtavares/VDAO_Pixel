import cv2
import os
import numpy as np
from src.dataset import VDAODataset
from torch.utils.data import DataLoader
from utils.srmc_outer_repr import srmc_outer_repr

video = 1
num_frames = 201
tarpath = '/home/luiz.tavares/Workspace/VDAO_Pixel/data/test/videos/tar'
refpath = '/home/luiz.tavares/Workspace/VDAO_Pixel/data/test/videos/ref'

cap_tar = cv2.VideoCapture(os.path.join(tarpath, '{0:02d}.avi'.format(video)))
cap_ref = cv2.VideoCapture(os.path.join(refpath, '{0:02d}.avi'.format(video)))

ref_vid = np.transpose(np.array(
    [cv2.cvtColor(cap_ref.read()[1], cv2.COLOR_RGB2GRAY)
     for ii in range(201)]), (1, 2, 0)) / 255
tar_vid = np.transpose(np.array(
    [cv2.cvtColor(cap_tar.read()[1], cv2.COLOR_RGB2GRAY)
     for ii in range(201)]), (1, 2, 0))/ 255

w, h = ref_vid.shape[1], ref_vid.shape[0]

ww = 280
hh = 150

params = {
    'windowSize': (hh, ww),
    'windowPos': ((h - hh) / 2, (w - ww) / 2),
    'transformType': '4P-HOMOGRAPHY',
    'display': 2,
    'stoppingDelta': 1e-4,
    'rho': 1.01,
    'inner_tol': 1e-4,
    'inner_maxIter': 5000,
    'lambda': 1e2,
    'DISPLAY_EVERY': 10
}

data = {
    'prevObj': np.inf,
    'numIterOuter': 0,
    'numIterInner': [],
    'W': {},
    'xi': {},
    'W_1': [],
    'E_1': [],
    'dt_1': [],
    'time': [],
    'rhos': [],
    'rerr': []
}

maxIter = 100
for k in range(data['numIterOuter'] + 1, maxIter):
    params['maxIter'] = k
    [Xr, Xt, W, E, xi, data] = srmc_outer_repr(ref_vid, tar_vid, params, data)
    if data['rerr'][k] < params['stoppingDelta']:
        break
