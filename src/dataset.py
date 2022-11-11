
import os
import glob
import sys
import torch 
import numpy as np
import pandas as pd
from PIL import Image

from src import PROJECT_DIR, VDAO_FRAMES_SHAPE
from src.resnet import Resnet50_Reduced
from torch.utils.data import DataLoader, Dataset, Subset
from src.config import fold_split
import torchvision.transforms as transforms
from src.utils import init_worker_random

# Definitions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
desired_output = (int(VDAO_FRAMES_SHAPE[0]/2), int(VDAO_FRAMES_SHAPE[1]/2))
init_worker_random()

# Define resnet
resnet = Resnet50_Reduced(device)
resize_transform    = transforms.Resize(desired_output)
normalize_transform = transforms.Normalize(mean=resnet.MEAN_IMAGENET, std=resnet.STD_IMAGENET)
to_tensor_transform = transforms.ToTensor()
transformations     = transforms.Compose([resize_transform, to_tensor_transform, normalize_transform])
sil_transform       = transforms.Compose([resize_transform, to_tensor_transform])

class VDAODataset(Dataset):

    def __init__(self,
                 fold  = 0,
                 split = 0,
                 data_path = '/nfs/proc/luiz.tavares/VDAO_Database/data/',
                 type = 'training',
                 alignment = 'warp',
                 transform = False,
                 patch=False):

        # Storing atributes
        self.fold = fold
        self.split = split
        self.type = type
        self.data_path = data_path
        self.transform = transform
        self.alignment = alignment
        
        # directories
        if self.type != 'training':
            fold_type = 'test'
        else:
            fold_type = 'training'
        self.data_path  = os.path.join(self.data_path, fold_type) 
        self.tar_path   = os.path.join(self.data_path, 'tar') 
        self.sil_path   = os.path.join(self.data_path, 'sil') 
        
        if self.alignment == 'warp':
            self.ref_path   = os.path.join(self.data_path, 'ref_warp') 
        elif self.alignment == 'geometric':
            self.ref_path   = os.path.join(self.data_path, 'ref_geo') 
        else:
            self.ref_path   = os.path.join(self.data_path, 'ref') 

        # Select set
        if self.type == 'training':
            if self.split > 0:
                fold_set = fold_split[self.fold][self.split][self.type]
            elif self.split == 0:
                fold_set = [self.fold]
            else:
                fold_set = [k for k in range(1,10) if k != self.fold]
        elif self.type == 'validation':
            if self.split > 0:
                fold_set = fold_split[self.fold][self.split][self.type]
            elif self.split == 0:
                fold_set = [self.fold]
            else:
                fold_set = [k for k in range(1,10) if k != self.fold]
        else:
            fold_set = [self.fold]
            
        # Load images
        self.ref_list, self.tar_list, self.sil_list = [], [], []
        self.align_df = pd.DataFrame()
        
        for fold_num in fold_set:
            # Get alignment files
            algn_file = os.path.join(PROJECT_DIR, 'data/alignment', 
                            'geometric_{0}_fold{1:02d}.csv'.format(fold_type, fold_num))
            algn = pd.read_csv(algn_file, index_col=0)
            if self.align_df.shape[0] == 0:
                self.align_df = algn
            else:
                self.align_df = pd.concat((self.align_df, algn),axis=0)
            
            
            # Get list of files
            self.tar_list.extend([os.path.join(self.tar_path, 'fold{0:02d}'.format(fold_num),'{0:04d}.png'.format(idx))
                                  for idx in list(self.align_df.index)])
            self.ref_list = [ii.replace(self.tar_path, self.ref_path) for ii in self.tar_list]
            self.sil_list = [ii.replace(self.tar_path, self.sil_path) for ii in self.tar_list]

        self.align_df.reset_index(drop=True, inplace=True)    

    def __len__(self):
        return len(self.ref_list)
    
    def _adjust_silhouette(self,sil):
        z = sil.astype(float)
        z[z<85]  = 0
        z[(z<=170) & (z>=85)]  = 0.5
        z[z>170] = 1

        return z

    def _get_info(self,idx):
        line = self.align_df.loc[idx]
        return {'file':line.target_file, 'frame':line.target_frame}

    def _crop(self, img, heigth, width):
        w_border = (img.shape[1] - width)//2
        h_border = (img.shape[0] - heigth)//2
        return img[h_border:h_border+heigth, w_border:w_border+width,...]
        
    
    def __getitem__(self, idx):
        
        ref_frame = np.array(Image.open(self.ref_list[idx]))
        tar_frame = np.array(Image.open(self.tar_list[idx]))
        sil_frame = np.array(Image.open(self.sil_list[idx]))

        # Crop
        if (self.alignment == 'geometric') | (self.alignment == 'warp'):
            tar_frame = self._crop(tar_frame, ref_frame.shape[0], ref_frame.shape[1])
            sil_frame = self._crop(sil_frame, ref_frame.shape[0], ref_frame.shape[1])   


        # Get silhouette 
        sil_frame = self._adjust_silhouette(sil_frame)
  
        if self.transform == True:

            # Transforming reference and target
            ref_frame = transformations(Image.fromarray(ref_frame))
            tar_frame = transformations(Image.fromarray(tar_frame))

            # Transforming and subsampling silhouette
            sil_frame = sil_transform(Image.fromarray(sil_frame))
            
        # else: 
        #     ref_frame = np.array(Image.open(self.ref_list[idx]))
        #     tar_frame = np.array(Image.open(self.tar_list[idx]))
               
        
        info = self._get_info(idx)

        return ref_frame, tar_frame, sil_frame, info

if __name__ == '__main__':

    dataset = VDAODataset(fold = 1, split = 0, type = 'validation',
                          alignment = 'elastic', transform = True)

    
    ref_img, tar_img, sil_img, info = next(iter(dataset))
    print(ref_img.shape, tar_img.shape, sil_img.shape)

    print('')
