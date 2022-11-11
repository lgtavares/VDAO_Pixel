import os
from PIL import Image
import pandas as pd
import numpy as np
from src import PROJECT_DIR
from src.config import fold_split
from src.video_loader import VideoLoader

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage import transform as tf


class VideoVDAODataset(Dataset):

    VDAO_FRAMES_SHAPE = [720, 1280, 3]

    def __init__(self,
                 fold_number,
                 split_number,
                 align_file=os.path.join(PROJECT_DIR,
                                         'dataset/alignment_vdao.csv'),
                 dataset_dir=os.path.join(PROJECT_DIR, 'dataset/VDAO/'),
                 type_dataset='train',
                 shuffle=True,
                 skip_frames=0,
                 object_only=False,
                 min_pixels=0,
                 video=0,
                 sil_frame_ss=4,
                 transformations=[],
                 geometric=False):

        # Asserting type
        assert type_dataset in [
            'training', 'validation', 'test'
        ], 'Argument type_dataset must be either \'training\', \
            \'validation\' or \'test\''

        # Storing parameters
        self.shuffle = shuffle
        self.fold_num = fold_number
        self.split_num = split_number
        self.align = align_file
        self.dataset_dir = dataset_dir
        self.skip = skip_frames
        self.type_dataset = type_dataset
        self.object_only = object_only
        self.min_pixels = min_pixels
        self.transformations = transformations
        self.sil_frame_ss = sil_frame_ss
        self.geometric = geometric
        self.video = video

        # Loading alignment file
        align_df = pd.read_csv(align_file, index_col=0, low_memory=False)

        # Select set
        if self.type_dataset != 'test':
            if self.split_num > 0:
                fold_set = fold_split[self.fold_num][self.split_num][
                    self.type_dataset]
            elif self.split_num == 0:
                fold_set = [self.fold_num]
            else:
                fold_set = [k for k in range(1, 10) if k != self.fold_num]
        else:
            fold_set = [self.fold_num]

        # get frames with these objects
        self.frames = align_df[align_df['target_obj'].isin(fold_set)]

        # Training / test split
        if self.type_dataset != 'test':
            self.frames = self.frames[not self.frames['test']]
        else:
            self.frames = self.frames[self.frames['test']]

        # self.frames = self.frames.drop_duplicates(['target_file',
        # 'target_frame'], keep= 'first')

        # Ballancing frames
        if self.object_only:
            self.frames = self.frames[self.frames['activity']]

        # filtering frames
        if self.type_dataset != 'test' and self.min_pixels != -1:
            self.frames = self.frames[
                self.frames['num_pixels'] >= self.min_pixels]

        # filtering video
        if self.video > 0:
            if self.type_dataset == 'test':
                self.frames = self.frames[self.frames['test_file'] ==
                                          self.video]
            else:
                self.frames = self.frames[self.frames['target_file'] ==
                                          self.video]

        # Skipping
        self.frames = self.frames.loc[::self.skip + 1, :]

        # Shuffle
        if self.shuffle:
            self.frames = self.frames.sample(frac=1)

        #  Resetting index
        self.frames = self.frames.reset_index(drop=True)

        # Database size
        self.num_samples = self.frames.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        # Frame data
        frame_data = self.frames.iloc[idx, :]
        ref_frame = frame_data['reference_frame']
        tar_frame = frame_data['target_frame']

        # Loading video
        self.vl = VideoLoader(video_idx=frame_data['target_file'],
                              database_dir=self.dataset_dir,
                              alignment=frame_data)

        # Getting frames
        ref_frame, tar_frame, sil_frame, _ = self.vl.get_frame(
            ref_frame, tar_frame)

        if self.geometric:

            # warping
            et_matrix = tf.EuclideanTransform(
                matrix=np.array(frame_data['homography'].split(','),
                                dtype=np.float).reshape(3, 3))
            ref_frame = tf.warp(ref_frame, et_matrix.inverse)
            ref_frame = (255 * ref_frame).astype('uint8')

            # cropping
            percentage = 0.05
            h, w = ref_frame.shape[0], ref_frame.shape[1]
            h_remove = int(h * percentage)
            w_remove = int(w * percentage)

            ref_frame = ref_frame[h_remove:h - h_remove,
                                  w_remove:w - w_remove, :]
            tar_frame = tar_frame[h_remove:h - h_remove,
                                  w_remove:w - w_remove, :]
            sil_frame = sil_frame[h_remove:h - h_remove,
                                  w_remove:w - w_remove, :]

        if self.transformations != []:

            # Transforming reference and target
            ref_frame = self.transformations(Image.fromarray(ref_frame))
            tar_frame = self.transformations(Image.fromarray(tar_frame))

            # Creating transformation for silhouette
            sil_transform = transforms.Compose(
                self.transformations.transforms[:2])

            # Transforming and subsampling silhouette
            sil_frame = sil_transform(
                (Image.fromarray(sil_frame).convert('L')))

        # Cropping sil_frame
        sil_frame = sil_frame[::self.sil_frame_ss, ::self.sil_frame_ss]

        # Important info [target_file,  object, frame, activity]
        if self.type_dataset != 'test':
            info = [
                frame_data['target_file'], frame_data['target_obj'],
                frame_data['target_frame'], frame_data['activity']
            ]
        else:
            info = [
                frame_data['test_file'], frame_data['target_obj'],
                frame_data['test_frame'], frame_data['activity']
            ]

        return ref_frame, tar_frame, sil_frame, info
