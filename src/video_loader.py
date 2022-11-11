import cv2
import sys
import random
import pandas as pd
import concurrent.futures

random_seed = 142
from src import PROJECT_DIR

class  VideoLoader:
    
    def __init__(self,
                 video_idx = 1,
                 database_dir = '/home/luiz.tavares/Workspace/VDAO_Pixel_old/dataset/VDAO/',
                 alignment = None):
        
        # setting variables
        self.video_idx       = video_idx
        self.database_dir    = database_dir
        self.alignment       = alignment
 
        
        self.executor     = concurrent.futures.ThreadPoolExecutor(max_workers=3)       
        
        self.ref_idx     = int(self.alignment['reference_file'])
        self.ref_frame   = int(self.alignment['reference_frame'])
        self.tar_frame   = int(self.alignment['target_frame'])

        self.ref_video_path = database_dir + 'ref/{0:02d}.avi'.format(self.ref_idx)
        self.tar_video_path = database_dir + 'obj/{0:02d}.avi'.format(self.video_idx)
        self.sil_video_path = database_dir + 'sil/{0:02d}.avi'.format(self.video_idx)
        
        # Create objet VideoCapture 
        self._cap_ref = cv2.VideoCapture(self.ref_video_path)
        self._cap_tar = cv2.VideoCapture(self.tar_video_path)
        self._cap_sil = cv2.VideoCapture(self.sil_video_path)
        
        
    def _get_frame(self, frame_number, video_type):
        
        if video_type == 'reference':
            # Como arquivo de anotação csv tem o frame 0 como o primeiro, obtém o frame normalmente
            self._cap_ref.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self._cap_ref.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        elif video_type == 'target':
            # Como arquivo de anotação csv tem o frame 0 como o primeiro, obtém o frame normalmente
            self._cap_tar.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self._cap_tar.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        elif video_type == 'silhouette':
            # Como arquivo de anotação csv tem o frame 0 como o primeiro, obtém o frame normalmente
            self._cap_sil.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self._cap_sil.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        return frame
    
    def __len__(self):
        return len(self.align_df)    

    def get_frame(self, ref_frame, tar_frame):
        

        ref_thread = self.executor.submit(self._get_frame, ref_frame, 'reference')
        tar_thread = self.executor.submit(self._get_frame, tar_frame, 'target')
        sil_thread = self.executor.submit(self._get_frame, tar_frame, 'silhouette')

        return (ref_thread.result(), tar_thread.result(), sil_thread.result(), self.alignment)