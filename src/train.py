"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""


from trainer import Solver 

import argparse

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help="path to real dataset")
    parser.add_argument('--syn_data_root', help="path to synthetic dataset")

    parser.add_argument('--root_dir', help="path to where outputs will be saved")
    parser.add_argument('--train_seq', type=list, default=[33])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--exp_name', type=str, default='depth_model')
    parser.add_argument('--num_epochs', type=int, default=15)

    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.train()
