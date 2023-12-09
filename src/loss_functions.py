"""
This repo is largely based on the code base of "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"
(https://github.com/koutilya-pnvr/SharinGAN) and heavily borrows from "EndoSLAM" (https://github.com/CapsuleEndoscope/EndoSLAM).

Edited by Anita Rau, a.rau.16@ucl.ac.uk, 2023
"""

from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp2
import matplotlib.pyplot as plt
import time
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from skimage.metrics import structural_similarity as ssim

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


def brightnes_equator(source, target, mask=None):
    def image_stats(image, mask=None):
        # compute the mean and standard deviation of each channel

        l = image[:,0,:,:]
        a = image[:,1,:,:]
        b = image[:,2,:,:]
        if mask == None:

            (lMean, lStd) = (torch.mean(torch.squeeze(l)), torch.std(torch.squeeze(l)))
            (aMean, aStd) = (torch.mean(torch.squeeze(a)), torch.std(torch.squeeze(a)))
            (bMean, bStd) = (torch.mean(torch.squeeze(b)), torch.std(torch.squeeze(b)))
        else:
            lMean = torch.mean(torch.squeeze(l).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))
            lStd = torch.std(torch.squeeze(l).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))
            aMean = torch.mean(torch.squeeze(a).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))
            aStd = torch.std(torch.squeeze(a).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))
            bMean = torch.mean(torch.squeeze(b).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))
            bStd = torch.std(torch.squeeze(b).reshape((256 * 256)) * mask.squeeze().reshape((256 * 256)))

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    def color_transfer(source, target, mask=None):
          # convert the images from the RGB to L*ab* color space, being
          # sure to utilizing the floating point data type (note: OpenCV
          # expects floats to be 32-bit, so use that instead of 64-bit)

          # compute color statistics for the source and target images
          (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source, mask)
          (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

          # subtract the means from the target image
          l = target[:,0,:,:]
          a = target[:,1,:,:]
          b = target[:,2,:,:]

          l = l - lMeanTar
          #print("after l",torch.isnan(l))
          a = a - aMeanTar
          b = b - bMeanTar
          # scale by the standard deviations
          l = (lStdTar / lStdSrc) * l
          a = (aStdTar / aStdSrc) * a
          b = (bStdTar / bStdSrc) * b
          # add in the source mean
          l = l + lMeanSrc
          a = a + aMeanSrc
          b = b + bMeanSrc
          transfer = torch.cat((l.unsqueeze(1),a.unsqueeze(1),b.unsqueeze(1)),1)
          #print(torch.isnan(transfer))
          return transfer

    # return the color transferred image
    transfered_image = color_transfer(target,source, mask)
    return transfered_image


def compute_pairwise_loss_orig(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask,
                          padding_mode):
    ref_img_warped, valid_mask, projected_depth, computed_depth, pose_mat = inverse_warp2(ref_img, tgt_depth, ref_depth, pose,  # inverse_warp2_orig
                                                                                intrinsic, padding_mode)

    ref_img_warped2 = brightnes_equator(ref_img_warped, tgt_img)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    #if with_auto_mask == True:
    #    auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1,keepdim=True)).float() * valid_mask
    #    valid_mask = auto_mask
    ssim_map = compute_ssim_loss(tgt_img, ref_img_warped2)
    if with_ssim == True:
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)
        diff_img = diff_img * weight_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)
    ssim_loss = mean_on_mask(ssim_map, valid_mask)  # 1 - ssim (so lower is better)
    tgt2 = tgt_img * valid_mask
    ref2 = ref_img_warped2 * valid_mask
    ssim_ = ssim(tgt2[0,:,:,:].permute((1,2,0)).cpu().detach().numpy(), ref2[0,:,:,:].permute((1,2,0)).cpu().detach().numpy(), multichannel=True)

    return reconstruction_loss, geometry_consistency_loss, ref_img_warped2, pose_mat, ssim_


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value



def compute_smooth_loss_orig(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp


        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss



def compute_photo_and_geometry_loss_orig(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode):
    photo_loss = 0
    geometry_loss = 0
    ssim_loss = 0
    warped_ims=[]
    intrinsics = intrinsics.cuda().float()

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):

            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='nearest')
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='nearest')

            photo_loss1, geometry_loss1, warped, pose_mat, ssim = compute_pairwise_loss_orig(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)

            photo_loss2, geometry_loss2, warped2, _, ssim2 = compute_pairwise_loss_orig(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
            if s==0:
                warped_ims.append([warped, warped2])
            photo_loss += ((photo_loss1 + photo_loss2)/2)
            geometry_loss += ((geometry_loss1 + geometry_loss2)/2)
            ssim_loss += (0.5 * (ssim + ssim2))

    return photo_loss, geometry_loss, warped_ims, pose_mat


def unity_quaternion_to_logq(q):
    u = q[:, -1]
    v = q[:, :-1]
    norm = torch.norm(v, dim=1, keepdim=True).clamp(min=1e-8)
    out = v * (torch.acos(torch.clamp(u, min=-1.0, max=1.0)).reshape(-1, 1) / norm)

    return out


def logq_to_unity_quaternion(w):
    norm = torch.norm(w)
    first = torch.cos(norm)
    second = w / norm * torch.sin(norm)

    return torch.cat([first, second])

def logq_to_quaternion(q):
    # return: quaternion with w, x, y, z
    #from geomap paper
    n = torch.norm(q, p=2, dim=1, keepdim=True)
    n = torch.clamp(n, min=1e-8)
    q = q * torch.sin(n)
    q = q / n
    q = torch.cat((torch.cos(n), q), dim=1)
    return q
    #return torch.cat([first, second])



class BalanceLosses(nn.Module):
    def __init__(self):
        super(BalanceLosses, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([0]).cuda(), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0]).cuda(), requires_grad=True)

    def forward(self, forward, backward):
        loss = forward*torch.exp(-self.beta) + self.beta + backward*torch.exp(-self.gamma) + self.gamma
        return loss

class LogQuatLossFixed(nn.Module):
    def __init__(self, criterion=nn.L1Loss()):
        super(LogQuatLossFixed, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor([-3]).cuda(), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor([0]).cuda(), requires_grad=True)
        self.criterion = criterion
        self.rot_criterion = nn.MSELoss()


    def forward(self, q_hat, q):
        # only use 1 direction for now
        t = q[:, :3]
        log_q = unity_quaternion_to_logq(q[:, 3:])
        t_hat = q_hat[:, :3]
        log_q_hat = q_hat[:, 3:]

        loss = self.criterion(t_hat, t) + self.rot_criterion(log_q_hat, log_q)*1000
        return loss

def quat2mat(q):
    # from github
    ''' Return Euler angles corresponding to quaternion `q`
    Parameters
    ----------
    q : 4 element sequence
       w, x, y, z of quaternion
    Returns
    -------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)
    Notes
    -----
    It's possible to reduce the amount of calculation a little, by
    combining parts of the ``quat2mat`` and ``mat2euler`` functions, but
    the reduction in computation is small, and the code repetition is
    large.
    '''
    # delayed import to avoid cyclic dependencies
    import nibabel.quaternions as nq
    return nq.quat2mat(q)



class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            #print('pred: ', prediction.shape)
            #print('targ: ', target_tensor.shape)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss