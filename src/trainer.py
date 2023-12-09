"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

import os
import numpy as np
import random
import torch
import torchvision
from tqdm import tqdm
from torch import nn
import torch.optim as Optim
import matplotlib.pyplot as plt


from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as tr
from tensorboardX import SummaryWriter

from networks import all_networks
from networks.da_net import Discriminator
from networks.all_networks import PoseResNet
from networks.all_networks import init_net


from Dataloaders.SimCol_dataloader import DepthToTensor
from Dataloaders.SimCol_dataloader import SimCol as syn_dataset
from Dataloaders.HCULB_dataloader import Hculb as real_dataset
import Dataloaders.transform as transf

from loss_functions import compute_photo_and_geometry_loss_orig, compute_smooth_loss_orig, SSIM


class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.opt.save = True
        self.specs = False
        self.seed = 0 # The famous Hardy-Ramanujan number
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Initialize networks
        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                  'PReLU', 'ResNet', 'kaiming', 0,
                                                  False, [0])
        self.netD = Discriminator(nout=1, last_layer_activation=False)
        init_net(self.netD)
    
        self.netT = all_networks.define_G(3, 1, 64, 4, 'instance',
                                            'ReLU', 'endo', 'kaiming', 0,
                                            False, [0], 0.1, alpha=100)


        self.netP = PoseResNet()
        self.netP.cuda()
        self.netG.cuda()
        self.netT.cuda()
        self.netD.cuda()

        # Initialize Loss
        self.netG_loss_fn = nn.L1Loss() #MSELoss()
        self.netD_loss_fn = nn.KLDivLoss()
        self.netT_loss_fn = nn.L1Loss()
        self.netG_loss_fn = self.netG_loss_fn.cuda()
        self.netD_loss_fn = self.netD_loss_fn.cuda()
        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Initialize Optimizers
        self.netD_optimizer = Optim.Adam(self.netD.parameters(), 0.00001, betas=(0.9, 0.999))
        self.netG_optimizer = Optim.Adam(self.netG.parameters(), 0.0001, betas=(0.9, 0.999))

        optim_params = [
            {'params': self.netT.parameters(), 'lr': 0.0001},
            {'params': self.netP.parameters(), 'lr': 0.0001}
        ]
        self.optimizer = torch.optim.Adam(optim_params,
                                     betas=(0.9, 0.999),
                                     weight_decay=0)

        # Training Configuration details
        self.batch_size = self.opt.batch_size  # was 8 in paper
        self.iteration = None
        self.START_ITER = 0
        self.flag = True
        self.best_a1 = 0.0
        self.i = np.eye(3).reshape((1,3,3))
        self.num_epochs = self.opt.num_epochs
        self.epoch = 0

        self.kr = 1
        self.kd = 1 
        self.kcritic = 5
        self.gamma = 10
        
        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])
        self.normalizer = tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        
        self.writer = SummaryWriter(os.path.join(self.opt.root_dir,'tensorboard_logs' + self.opt.exp_name))
        self.save_model_dir = 'saved_models/' + self.opt.exp_name 

        # Initialize Data
        self.get_training_data()
        self.get_training_dataloaders()
        self.get_validation_data()

        self.num_synth = self.syn_loader.__len__()
        self.num_real = self.real_loader.__len__()
        self.total_iterations = self.num_synth * self.num_epochs
        self.log_freq = self.num_synth * 3

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

    def get_training_data(self):

        self.syn_loader = DataLoader(syn_dataset(data_root=self.opt.syn_data_root, train=True, specs=self.specs, norm=False), batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)
        self.real_loader = DataLoader(real_dataset(self.opt.train_seq, data_root=self.opt.data_root, train=True, frames_apart=5, norm=False), batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=True)

    def get_training_dataloaders(self):
        self.syn_iter = self.loop_iter(self.syn_loader)
        self.real_iter = self.loop_iter(self.real_loader)

    def get_validation_data(self):

        self.syn_val_loader = DataLoader(syn_dataset(data_root=self.opt.syn_data_root, train=False, specs=self.specs, norm=False, gap=10), batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)
        self.real_val_loader = DataLoader(real_dataset(self.opt.train_seq, data_root=self.opt.data_root, train=False, frames_apart=5, norm=False, gap=10), batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=True)

            
    def load_prev_model(self, model_status='latest'):
        saved_model = os.path.join(self.opt.root_dir, self.saved_models_dir, 'Depth_Estimator-da_tmp.pth.tar')
        print('loading model...', saved_model)
        model_state = torch.load(saved_model)
        self.netG.load_state_dict(model_state['netG_state_dict'])
        self.netT.load_state_dict(model_state['netT_state_dict'])
        self.netP.load_state_dict(model_state['netP_state_dict'])

        self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
        self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
        self.netP_optimizer.load_state_dict(model_state['netP_optimizer'])

        for i,disc in enumerate(self.netD):
            disc.load_state_dict(model_state['netD'+str(i)+'_state_dict'])
            self.netD_optimizer[i].load_state_dict(model_state['netD'+str(i)+'_optimizer_state_dict'])

        self.START_ITER = model_state['iteration']+1
        return True

    def save_model(self, model_status='latest'):
        print('\nsaving model to ', self.save_model_dir)
        if not os.path.exists(os.path.join(self.opt.root_dir, self.save_model_dir)):
            os.mkdir(os.path.join(self.opt.root_dir, self.save_model_dir))

        dict1 = {
            'iteration': self.iteration,
            'netG_state_dict': self.netG.state_dict(),
            'netT_state_dict': self.netT.state_dict(),
            'netP_state_dict': self.netP.state_dict(),
            'netD_state_dict': self.netD.state_dict(),
            'netG_optimizer': self.netG_optimizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'netD_optimizer': self.netD_optimizer.state_dict(),
        }

        final_dict = dict1.copy()
        torch.save(final_dict, os.path.join(self.opt.root_dir, self.save_model_dir, 'DepthModel'+str(self.iteration)+'.pth.tar'))
        # os.system('mv '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator-da_tmp.pth.tar')+' '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_WI_geom_bicubic_da-'+model_status+'.pth.tar'))

    def get_syn_data(self):
        self.syn_image_spec, self.syn_label, self.syn_image = next(self.syn_iter)
        self.syn_image, self.syn_label, self.syn_image_spec = self.syn_image.cuda(), self.syn_label.cuda(), self.syn_image_spec.cuda()
        self.syn_label_scales = self.scale_pyramid(self.syn_label, 5-1)
        self.syn_image_normed = self.normalizer(self.syn_image)

    def get_real_data(self):
        self.real_image, self.real_right_image, self.fb, self.K = next(self.real_iter)
        self.real_image = Variable(self.real_image.cuda())
        self.real_right_image = [Variable(im.cuda()) for im in self.real_right_image]

        self.real_image_normed = self.normalizer(self.real_image).cuda()
        self.real_right_image_normed = [self.normalizer(im).cuda() for im in self.real_right_image]

        self.real_image_scales = self.scale_pyramid(self.real_image, 5-1)
        self.real_right_image_scales = [self.scale_pyramid(im, 5-1) for im in self.real_right_image]
        
    def gradient_penalty(self, model, h_s, h_t):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        batch_size =min(h_s.size(0), h_t.size(0))
        h_s = h_s[:batch_size]
        h_t = h_t[:batch_size]
        size = len(h_s.shape)
        alpha = torch.rand(batch_size)#, 1, 1, 1)
        for ki in range(1,size):
            alpha = alpha.unsqueeze(ki)
        alpha = alpha.expand_as(h_s)
        alpha = alpha.cuda()
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
        interpolates = Variable(interpolates.cuda(), requires_grad=True)
        preds = model(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                            grad_outputs=torch.ones_like(preds).cuda(),
                            retain_graph=True, create_graph=True)[0]
        gradients = gradients.view(batch_size,-1) 
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty
    
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]

        s = img.size()

        h = s[2]
        w = s[3]

        for i in range(1, num_scales):
            ratio = 2**i
            nh = h // ratio
            nw = w // ratio
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=True)
            scaled_imgs.append(scaled_img)

        return scaled_imgs
    
    def gradient_x(self,img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx


    def gradient_y(self,img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy

    def get_smooth_weight(self, depths, Images, num_scales):

        depth_gradient_x = [self.gradient_x(d) for d in depths]
        depth_gradient_y = [self.gradient_y(d) for d in depths]

        Image_gradient_x = [self.gradient_x(img) for img in Images]
        Image_gradient_y = [self.gradient_y(img) for img in Images]

        weight_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_x]
        weight_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_y]

        smoothness_x = [depth_gradient_x[i] * weight_x[i] for i in range(num_scales)]
        smoothness_y = [depth_gradient_y[i] * weight_y[i] for i in range(num_scales)]

        loss_x = [torch.mean(torch.abs(smoothness_x[i]))/2**i for i in range(num_scales)]
        loss_y = [torch.mean(torch.abs(smoothness_y[i]))/2**i for i in range(num_scales)]

        return sum(loss_x+loss_y)
    
    def reset_netD_grad(self, i=None):
        if i==None:
            self.netD_optimizer.zero_grad()
        else:
            for idx, disc_op in enumerate(self.netD):
                if idx==i:
                    continue
                else:
                    disc_op.zero_grad()

    def reset_grad(self, exclude=None):
        if(exclude==None):
            self.netG_optimizer.zero_grad()
            self.netD_optimizer.zero_grad()
            self.netT_optimizer.zero_grad()
            self.netP_optimizer.zero_grad()
        elif(exclude=='netG'):
            self.netD_optimizer.zero_grad()
            self.netT_optimizer.zero_grad()
            self.netP_optimizer.zero_grad()
        elif(exclude=='netD'):
            self.netG_optimizer.zero_grad()
            self.netT_optimizer.zero_grad()
            self.netP_optimizer.zero_grad()
        elif(exclude=='netT'):
            self.netG_optimizer.zero_grad()
            self.netD_optimizer.zero_grad()
            self.netP_optimizer.zero_grad()
        elif(exclude=='netGT'):
            self.netD_optimizer.zero_grad()
            self.netP_optimizer.zero_grad()
        elif(exclude=='netTP'):
            self.netG_optimizer.zero_grad()
            self.netD_optimizer.zero_grad()

    def forward_netD(self, mode='gen'):
        self.D_syn = self.netD((self.syn_recon_image + 1) / 2)
        self.D_real = self.netD((self.real_recon_image + 1) / 2)
    
    def loss_from_disc(self, mode='gen'):
        self.just_adv_loss = self.D_syn.mean() - self.D_real.mean()
        if mode == 'disc':
            self.just_adv_loss = -1* self.just_adv_loss
        
    def set_requires_grad(self, model, mode=False):
        for param in model.parameters():
            param.requires_grad = mode 

    def update_netG(self):
        
        self.set_requires_grad(self.netT, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netP, False)

        self.real_features, self.real_recon_image = self.netG(self.real_image)
        self.syn_features, self.syn_recon_image = self.netG(self.syn_image_spec)
        self.forward_netD()
        self.loss_from_disc()

        mask = (self.real_image.sum(1) < 2.9).float()  # Mask out specularities
        real_reconstruction_loss = torch.mean((torch.abs((self.real_recon_image+1)/2 - self.real_image)).mean(1) * mask)
        syn_reconstruction_loss = self.netG_loss_fn((self.syn_recon_image+1)/2, self.syn_image)
        self.recon_loss = real_reconstruction_loss + syn_reconstruction_loss
        
        real_depth = self.netT(self.normalizer((self.real_recon_image + 1)/2))
        syn_depth = self.netT(self.normalizer((self.syn_recon_image + 1) / 2))

        syn_task_loss = 0.0
        for (lab_fake_i, lab_real_i) in zip(syn_depth[:3], self.syn_label_scales[:3]):
            syn_task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)

        real_right_depth = [self.netT(self.normalizer((self.netG(im)[1]+1)/2)) for im in self.real_right_image]
        pose = [self.netP(self.real_image_normed, im) for im in self.real_right_image_normed]
        pose_inv = [self.netP(im, self.real_image_normed) for im in self.real_right_image_normed]
        loss_1, loss_3, warped, _ = compute_photo_and_geometry_loss_orig(self.real_image_normed, self.real_right_image_normed, self.K, real_depth, real_right_depth, pose, pose_inv, 3, 0, 1, 1, 'zeros')
        loss_2 = compute_smooth_loss_orig(real_depth, self.real_image_normed, real_right_depth, self.real_right_image_normed)

        real_task_loss = (1 * loss_1 + 0.5 * loss_3 + 0.1 * loss_2)

        real_size = len(real_depth)
        gradient_smooth_loss = self.get_smooth_weight(real_depth[:-1], self.real_image_scales, real_size-1)


        self.netG_loss = 1 * self.just_adv_loss + 0.5 * (100 * syn_task_loss + 1 * real_task_loss) + (0.01*gradient_smooth_loss)
        self.netG_loss += 10 * self.recon_loss
        
        self.optimizer.zero_grad()
        self.netG_optimizer.zero_grad()
        self.netD_optimizer.zero_grad()
        self.netG_loss.backward()
        self.optimizer.zero_grad()
        self.netD_optimizer.zero_grad()
        self.netG_optimizer.step()

        self.set_requires_grad(self.netT, True)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netP, True)



    def update_netT(self):
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netD, False)

        syn_recon = self.normalizer((self.netG(self.syn_image)[1]+1)/2)

        syn_depth = self.netT(syn_recon)
        task_loss = 0.0
        for (lab_fake_i, lab_real_i) in zip(syn_depth[:3], self.syn_label_scales[:3]):
            task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)
        self.syn_depth_loss = task_loss


        real_recon = self.normalizer((self.netG(self.real_image)[1]+1)/2)
        real_right_recon = [self.normalizer((self.netG(im)[1]+1)/2) for im in self.real_right_image]

        real_depth = self.netT(real_recon)
        real_right_depth = [self.netT(im) for im in real_right_recon]
        pose = [self.netP(real_recon, im) for im in real_right_recon]
        pose_inv = [self.netP(im, real_recon) for im in real_right_recon]
        loss_1, loss_3, warped, _ = compute_photo_and_geometry_loss_orig(self.real_image, self.real_right_image, self.K, real_depth, real_right_depth, pose, pose_inv, 3, 0, 1, 1, 'zeros')
        loss_2 = compute_smooth_loss_orig(real_depth, self.real_image, real_right_depth, self.real_right_image)

        real_size = len(real_depth)
        self.gradient_smooth_loss = self.get_smooth_weight(real_depth[:-1], self.real_image_scales, real_size - 1)

        if self.iteration % 200 == 0:
            with torch.no_grad():

                i=0
                fig_real_warped, ax2 = plt.subplots(2, 3, figsize=(7, 3))
                ax2[0,0].imshow((self.real_image[i]).clip(0,1).permute((1,2,0)).detach().cpu().numpy())
                ax2[0,1].imshow((self.real_right_image[1][i]).clip(0,1).permute((1,2,0)).detach().cpu().numpy())
                ax2[0,2].imshow((warped[1][0][i]).clip(0, 1).permute((1,2,0)).detach().cpu().numpy())
                ax2[1,0].imshow((real_depth[0][i,0,:,:]).detach().cpu().numpy())
                ax2[1,1].imshow((real_right_depth[1][0][i,0,:,:]).detach().cpu().numpy())
                ax2[1,2].imshow((warped[1][1][i]).clip(0, 1).permute((1,2,0)).detach().cpu().numpy())

                fig_real_warped.suptitle(str(pose[1][i].detach().cpu().numpy()))
                self.writer.add_figure('Warped real', fig_real_warped, self.iteration)

        self.warp_loss = (1 * loss_1 + 0.5 * loss_3 + 0.1 * loss_2)

        self.writer.add_scalar('photometric_error', loss_1.item(), self.iteration)
        self.writer.add_scalar('disparity_smoothness_loss', loss_2.item(), self.iteration)
        self.writer.add_scalar('geometry_consistency_loss', loss_3.item(), self.iteration)
        self.writer.add_scalar('syn_depthpyramid_loss', self.syn_depth_loss.item(), self.iteration)
        self.writer.add_scalar('recon_loss', self.recon_loss.item(), self.iteration)

        self.netT_loss = 0.5 * (1 * self.warp_loss + 100 * self.syn_depth_loss) + 0.01 * self.gradient_smooth_loss

        self.optimizer.zero_grad()
        self.netG_optimizer.zero_grad()
        self.netT_loss.backward()
        self.netG_optimizer.zero_grad()
        self.optimizer.step()
        self.set_requires_grad(self.netG, True)
        self.set_requires_grad(self.netD, True)


    def update_netD(self):
        
        self.set_requires_grad(self.netG, False)

        with torch.no_grad():
            self.syn_features, self.syn_recon_image  = self.netG(self.syn_image)
            self.real_features, self.real_recon_image = self.netG(self.real_image)

        for _ in range(self.kcritic):
            self.forward_netD(mode='disc')
            self.loss_from_disc(mode='disc')
            
            gp = self.gradient_penalty(self.netD, self.syn_recon_image, self.real_recon_image)
            self.netD_loss = self.just_adv_loss + self.gamma*gp

            self.optimizer.zero_grad()
            self.netG_optimizer.zero_grad()
            self.netD_optimizer.zero_grad()
            self.netD_loss.backward()
            self.optimizer.zero_grad()
            self.netG_optimizer.zero_grad()
            self.netD_optimizer.step()

        self.set_requires_grad(self.netG, True)



    def train(self):
        self.get_syn_data()
        for self.iteration in tqdm(range(self.START_ITER, self.total_iterations)):

            self.get_real_data()
            self.get_syn_data()

            self.update_netD()

            for i in range(self.kr):
                self.update_netG()

            self.update_netT()


            self.writer.add_scalar('1) Total Generator loss', self.netG_loss, self.iteration)
            self.writer.add_scalar('2) Total Discriminator loss', self.netD_loss, self.iteration)
            self.writer.add_scalar('3) Total Depth Regressor loss', self.netT_loss, self.iteration)
            self.writer.add_scalar('4) Warp loss', self.warp_loss, self.iteration)
            self.writer.add_scalar('5) Gradient smooth loss ', self.gradient_smooth_loss, self.iteration)


            if self.iteration % self.log_freq == 0: 
                # Validation and saving models
                if self.opt.save:
                    self.save_model(model_status='latest')
                print('validating')
                self.Validate()
                print('training')
                self.epoch += 1
            elif self.iteration % 200 == 0:
                self.log()

                
        self.writer.close()



    def Validate(self):
        self.netG.eval()
        self.netT.eval()

        depth_errs = []
        with torch.no_grad():
            idx = np.random.randint(self.real_val_loader.__len__())
            for i, (data, depth_filenames) in enumerate(self.real_val_loader):
                real_val_image = data['left_img'].cuda()
                _, real_recon_image = self.netG(real_val_image)
                real_recon_image = (real_recon_image +1) /2

                depth = self.netT(self.normalizer(real_recon_image))

                if i == idx:
                    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3))
                    ax2[0].imshow(depth[0, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
                    ax2[1].imshow(real_val_image[0, :, :, :].permute((1,2,0)).cpu().numpy())
                    ax2[2].imshow(real_recon_image[0, :, :, :].permute((1, 2, 0)).cpu().numpy())
                    self.writer.add_figure('Validation Depth Real', fig2, self.epoch)


            idx = np.random.randint(10)
            for i, (syn_image, syn_depth, _) in enumerate(self.syn_val_loader):
                _, syn_recon_image = self.netG(syn_image)
                syn_recon_image = (syn_recon_image +1) /2
                pred_syn_depth = self.netT(self.normalizer(syn_recon_image))

                if i == idx:
                    fig, ax = plt.subplots(1, 4, figsize=(13, 3))
                    ax[0].imshow(syn_depth[0,0,:,:].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
                    ax[1].imshow(pred_syn_depth[0,0,:,:].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
                    ax[2].imshow(syn_image[0, :, :, :].permute((1, 2, 0)).cpu().numpy())
                    ax[3].imshow(syn_recon_image[0, :, :, :].permute((1, 2, 0)).cpu().numpy())

                    self.writer.add_figure('Validation Depth Synth', fig, self.epoch)

                    break


                depth_errs.append(self.netT_loss_fn(syn_depth.cuda(), pred_syn_depth).cpu().numpy())

            self.writer.add_scalar('Val L1 loss', np.mean(depth_errs), self.epoch)



        self.netG.train()
        self.netT.train()


    def log(self):
        self.netG.eval()
        self.netT.eval()

        with torch.no_grad():
            _, real_recon_image = self.netG(self.real_image)
            real_recon_image = (real_recon_image + 1) / 2
            depth = self.netT(self.normalizer(real_recon_image))

            fig2, ax2 = plt.subplots(1, 2, figsize=(7, 3))
            ax2[0].imshow(depth[0, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
            ax2[1].imshow(depth[1, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')

            self.writer.add_image('Real Images',
                                  torchvision.utils.make_grid(self.real_image, nrow=4),
                                  self.iteration)
            self.writer.add_figure('Predicted Depth', fig2, self.iteration)
            self.writer.add_image('Real Translated Images',
                                  torchvision.utils.make_grid(real_recon_image, nrow=4),
                                  self.iteration)


            _, syn_recon_image = self.netG(self.syn_image)
            syn_recon_image = (syn_recon_image + 1) / 2

            pred_syn_depth = self.netT(self.normalizer(syn_recon_image))

            fig, ax = plt.subplots(1, 2, figsize=(7, 3))
            ax[0].imshow(pred_syn_depth[0, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
            ax[1].imshow(pred_syn_depth[1, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
            fig2, ax2 = plt.subplots(1, 2, figsize=(7, 3))
            ax2[0].imshow(self.syn_label[0, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')
            ax2[1].imshow(self.syn_label[1, 0, :, :].cpu().numpy(), vmin=0, vmax=0.8, cmap='viridis_r')

            self.writer.add_image('Synth Translated Images',
                                  torchvision.utils.make_grid(syn_recon_image, nrow=4),
                                  self.iteration)
            self.writer.add_image('Synth Orig Images',
                                  torchvision.utils.make_grid(self.syn_image_spec, nrow=4),
                                  self.iteration)
            self.writer.add_figure('Predicted Synth Depth', fig, self.iteration)
            self.writer.add_figure('GT Synth Depth', fig2, self.iteration)


        self.netG.train()
        self.netT.train()