"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import os
import numpy as np
import random
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as tr
from networks import all_networks
import matplotlib.pyplot as plt
from Dataloaders.HCULB_dataloader import Hculb as real_dataset


class Solver():
    def __init__(self, opt):

        self.opt = opt
        self.seed = 1729 # The famous Hardy-Ramanujan number
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                          'PReLU', 'ResNet', 'kaiming', 0,
                                          False, [0])


        self.netT = all_networks.define_G(3, 1, 64, 4, 'instance',
                                          'ReLU', 'endo', 'kaiming', 0,
                                          False, [0], 0.1, alpha=100)

        self.netG.cuda()
        self.netT.cuda()

        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.get_validation_data()


    def get_validation_data(self):
        self.real_val_loader = DataLoader(real_dataset(self.opt.test_seq, data_root=self.opt.data_root, train=False, frames_apart=1, norm=False, gap=1, shifts=[1]), batch_size=1, shuffle=False, num_workers=2, drop_last=True)
      

    def test(self):
        print('Loading the trained model from {}'.format(self.opt.model_path))
        model_state = torch.load(self.opt.model_path)

        self.netT.load_state_dict(model_state['netT_state_dict'])
        self.netG.load_state_dict(model_state['netG_state_dict'])
        self.name = 'replicate_results'
        self.Validate()


    def Validate(self):
        self.netG.eval()
        self.netT.eval()
        path = os.path.join(self.opt.output_path, self.name)
        if not os.path.exists(path):
            os.mkdir(path)

        with torch.no_grad():
            for i, (data, depth_filenames) in enumerate(self.real_val_loader):
                real_val_image = data['left_img'].cuda()
                _, real_recon_image = self.netG(real_val_image)
                real_recon_image = (real_recon_image + 1) / 2
                depth = self.netT(self.normalizer(real_recon_image))

                fig2, ax2 = plt.subplots(1, 3, figsize=(9, 3))
                ax2[0].imshow(depth[0, 0, :, :].cpu().numpy(), cmap='viridis_r')
                ax2[0].set_title('Predicted Depth')
                ax2[1].imshow(real_val_image[0, :, :, :].permute((1,2,0)).cpu().numpy())
                ax2[1].set_title('Input Image')
                ax2[2].imshow(real_recon_image[0, :, :, :].permute((1, 2, 0)).cpu().numpy())
                ax2[2].set_title('Translated Image')
                plt.subplots_adjust(wspace=0.03, hspace=0.03)
                [axi.set_xticks([]) for axi in ax2.ravel()]
                [axi.set_yticks([]) for axi in ax2.ravel()]
                fig2.savefig(path + '/' + depth_filenames[0].split('/')[-1],  bbox_inches='tight')
                plt.close(fig2)


def get_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_path', default='outputs/')
    parser.add_argument('--test_seq', type=list, default=[33])
    parser.add_argument('--data_root')
    parser.add_argument('--model_path', default='trained_models/DepthModel.pth.tar')
    opt = parser.parse_args()
    return opt


if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.test()