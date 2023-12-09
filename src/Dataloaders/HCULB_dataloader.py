"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""


import torch.utils.data as data
import numpy as np
from path import Path
import random
from PIL import Image
from PIL import ImageFilter
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import torchvision.transforms.functional as TF
import torch
import cv2


def load_as_float(path):
    return Image.open(path)


class Hculb(data.Dataset):
    def __init__(self, vid_ids, data_root='/path/to/data/', seed=None, train=True, sequence_length=3, custom_transform=None, skip_frames=1,
                 resize=True, frames_apart=1, gap=1, offset=0, im_size=256, depth=False, norm=True, shifts=[-1,1], pairs=True):
        np.random.seed(seed)
        random.seed(seed)
        self.custom_transform = custom_transform
        self.k = skip_frames
        self.frames_apart = frames_apart
        self.train = train
        self.data_root = data_root
        self.gap = gap
        self.offset = offset
        self.depth = depth
        self.norm = norm
        self.shifts = shifts
        self.pairs = pairs
        self.newK = np.array([[ 58.30109866849495,0.,139.57619298432274],[0.,58.32320920632603, 135.1197128002675],[0,0,1]]) #undistorted intrinsics

        self.crawl_folders(sequence_length)

        self.to_norm_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.to_tensor = transforms.ToTensor()

        if self.train:
            self.resizer = transforms.Compose([transforms.Resize((256, 256)), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])
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



    def rescale_matrix(self, intrinsics_matrix):
        intrinsics_matrix[0, :] = intrinsics_matrix[0, :] * self.scale_x
        intrinsics_matrix[1, :] = intrinsics_matrix[1, :] * self.scale_y
        return intrinsics_matrix


    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length - 1) // 2

        imgs = []
        col_vid = 33 
        if self.train:
            seqs = [2, 3, 7, 9, 10, 8]
        else:
            seqs = [4, 5, 6, 15, 21]
        num_ims = 0
        patient= [33]

        for col_vid in patient:
            for seq in seqs:
                imgs = []
                for i, scene in enumerate(open(self.data_root+str(col_vid)+'/cluster_list/' + str(seq) + '.txt','r')):
                    if i % 1 == 0:
                        imgs.append(Path(self.data_root+str(col_vid)+'_undist/img_train/' + scene.strip(' \n')))
                        num_ims += 1


                intrinsics = np.eye(3)
                imgs = sorted(imgs)


                for i in range(self.offset + demi_length * self.k + self.frames_apart - 1,
                               len(imgs) - demi_length * self.k - self.frames_apart + 1,
                               self.gap):
                    sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                    for j in self.shifts:
                        sample['ref_imgs'].append(imgs[i + j * self.frames_apart])
                    sequence_set.append(sample)

                if self.train:
                    for i in range(demi_length * self.k + self.frames_apart - 1 + 2,
                                   len(imgs) - demi_length * self.k - self.frames_apart + 1 - 2):
                        sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                        for j in self.shifts:
                            sample['ref_imgs'].append(imgs[i + j * (self.frames_apart + 1)])

                        sequence_set.append(sample)

                        sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                        for j in self.shifts:
                            sample['ref_imgs'].append(imgs[i + j * (self.frames_apart - 1)])

                        sequence_set.append(sample)


        self.samples = sequence_set


    def __getitem__(self, index):
        sample = self.samples[index]

        tgt_img = Image.open(sample['tgt'])
        if self.pairs:
            ref_imgs = [Image.open(ref_img) for ref_img in sample['ref_imgs']]

        if self.norm:
            tgt_img = self.to_norm_tensor(np.array(tgt_img)[:,:,:3])
            if self.pairs:
                ref_imgs = [self.to_norm_tensor(np.array(ref_img)[:, :, :3]) for ref_img in ref_imgs]
        else:
            tgt_img = self.to_tensor(np.array(tgt_img))[:3, :, :]
            if self.pairs:
                ref_imgs = [self.to_tensor(np.array(ref_img))[:3, :, :] for ref_img in ref_imgs]



        if self.train:
            if self.pairs:
                return tgt_img, ref_imgs, 1., self.newK
            else:
                return tgt_img, [], 1., self.newK
        else:
            data = {}
            data['left_img'] = tgt_img
            data['right_img'] = ref_imgs[0]
            data['fb'] = self.newK

            return data, sample['tgt']



    def __len__(self):
        if self.train:
            return len(self.samples)
        else:
            return len(self.samples)
