"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import torch.utils.data as data
import numpy as np
from path import Path
import random
from PIL import Image, ImageDraw
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import torch


def load_as_float(path):
    return Image.open(path)

class DepthToTensor(object):
    def __call__(self, arr_input):
        tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1]))).float()
        return tensors


class SimCol(data.Dataset):
    def __init__(self, data_root='/path/to/synthetic/data', seed=None, train=True, sequence_length=3, custom_transform=None, skip_frames=1,
                 dataset='S', resize=True, frames_apart=1, gap=1, offset=0,
                 train_file='train_file.txt', val_file='val_file.txt', im_size=256, depth=True, specs=True,norm=True, shifts = [-1, 1]):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(data_root)
        scene_list_path = self.root / train_file if train else self.root / val_file
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.custom_transform = custom_transform
        self.dataset = dataset
        self.k = skip_frames
        self.frames_apart = frames_apart
        self.train = train
        self.gap = gap
        self.offset = offset
        self.depth = depth

        self.crawl_folders(sequence_length, datasetname=dataset)

        self.to_norm_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.to_tensor = transforms.ToTensor()

        if self.train:
            self.resizer = transforms.Compose([transforms.Resize((256, 256))])
        else:
            self.resizer = transforms.Resize((256, 256))

        self.depth_resizer = transforms.Resize((256, 256))
        self.resize = resize
        self.scale_x = im_size / 475
        self.scale_y = im_size / 475
        self.Tx = np.eye(4)
        self.Tx[0, 0] = -1
        self.Ty = np.eye(4)
        self.Ty[1, 1] = -1
        self.specs = specs
        self.norm = norm



    def rescale_matrix(self, intrinsics_matrix):
        intrinsics_matrix[0, :] = intrinsics_matrix[0, :] * self.scale_x
        intrinsics_matrix[1, :] = intrinsics_matrix[1, :] * self.scale_y
        return intrinsics_matrix


    def crawl_folders(self, sequence_length, datasetname):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        for scene in self.scenes:
            print(scene)
            intrinsics = np.genfromtxt(scene / 'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('F*.png'))
            print('get: ', scene.replace('DATA','data').split('_'+datasetname)[1][:-1])

            if len(imgs) < sequence_length:
                continue

            if self.train:

                for i in range(demi_length * self.k + self.frames_apart - 1 + 2,
                               len(imgs) - demi_length * self.k - self.frames_apart + 1 - 2):
                # This should just be: for i in range(len(imgs)):, but the above is what we used in the paper. 
                # The above will just remove a couple of images at the beginning and end for warping, but we don't 
                # need warping for the synthetic images.)

                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                    sequence_set.append(sample)


            else:
                for i in range(self.offset + demi_length * self.k - 1,
                               len(imgs) - demi_length * self.k - self.frames_apart + 1,
                               self.gap):
                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': [], 'ref_poses': []}
                    sequence_set.append(sample)

        self.samples = sequence_set


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_depth_path = str(sample['tgt']).replace('FrameBuffer', 'Depth').replace('_Frames', '')

        if self.resize:
            tgt_img = self.resizer(Image.open(sample['tgt']).convert('RGB'))

        else:
            tgt_img = Image.open(sample['tgt']).convert('RGB')

        if self.specs:
            spec_im = tgt_img.copy()
            r = np.random.randint(0, 20, 1)  # Random number of specs
            draw = ImageDraw.Draw(spec_im)

            for i in range(r[0]):
                xy = np.random.rand(2) * 256  # random position of spec, 256 is image size
                width = np.random.rand() * 25  # random width of spec
                height = np.random.rand() * 25  # random height of spec
                draw.ellipse([xy[0], xy[1], xy[0] + width, xy[1] + height], fill='white', outline=None)
            if self.norm:
                spec_im = self.to_norm_tensor(np.array(spec_im))[:3, :, :]
            else:
                spec_im = self.to_tensor(np.array(spec_im))[:3, :, :]

        if self.norm:
            tgt_img = self.to_norm_tensor(np.array(tgt_img))[:3, :, :]
        else:
            tgt_img = self.to_tensor(np.array(tgt_img))[:3, :, :]


        if self.depth:
            if self.resize:
                tgt_gt_depth = self.depth_resizer(Image.open(tgt_depth_path))

            else:
                tgt_gt_depth = Image.open(tgt_depth_path)

            tgt_gt_depth = self.to_tensor((tgt_gt_depth)) / 65000.


        if self.specs:
            return spec_im, tgt_gt_depth, tgt_img
        else:
            return tgt_img, tgt_gt_depth, tgt_img



    def __len__(self):
        return len(self.samples)
