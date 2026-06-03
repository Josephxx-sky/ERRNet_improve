import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
from torch import autograd
import torchvision.models as models
import util.util as util
from models.vgg import Vgg19
from torch.autograd import Function
from models.CX import CX_loss

###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class CharbonnierLoss(nn.Module):
    """Charbonnier (smooth L1) loss: sqrt((x - y)^2 + eps^2).
    相比 MSE 对离群点更鲁棒，通常能缓解输出模糊。
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, predict, target):
        diff = predict - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class FFTLoss(nn.Module):
    """Amplitude-spectrum loss in frequency domain.
    惩罚预测与目标在傅里叶振幅谱上的差异，与空域损失正交，
    有助于保留高频细节、缓解空域 MSE 导致的过度平滑。
    """
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, predict, target):
        # Real FFT over spatial dims, normalized (ortho)
        pred_fft = torch.fft.rfft2(predict, norm='ortho')
        tgt_fft = torch.fft.rfft2(target, norm='ortho')
        # Penalize amplitude (abs) differences; phase is harder to supervise
        pred_amp = torch.abs(pred_fft)
        tgt_amp = torch.abs(tgt_fft)
        return self.criterion(pred_amp, tgt_amp)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss: 1 - SSIM."""
    def __init__(self, window_size=11, channel=3, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.register_buffer("window", self._create_window(window_size, channel))

    def _gaussian(self, window_size, sigma=1.5):
        gauss = torch.tensor(
            [np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
            dtype=torch.float32
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1d_window = self._gaussian(window_size).unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2d_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, predict, target):
        channel = predict.size(1)
        if channel != self.channel or self.window.device != predict.device or self.window.dtype != predict.dtype:
            self.channel = channel
            self.window = self._create_window(self.window_size, channel).to(device=predict.device, dtype=predict.dtype)

        mu1 = F.conv2d(predict, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(predict * predict, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(predict * target, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        ssim_value = ssim_map.mean() if self.size_average else ssim_map.mean(dim=(1, 2, 3))
        return 1.0 - ssim_value


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        device = next(self.vgg.parameters()).device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


class CXLoss(VGGLoss):
    # Contextual Loss from
    # https://arxiv.org/abs/1803.02077
    def __init__(self, vgg=None, weights=None, indices=None, criterions=None):        
        super(CXLoss, self).__init__(vgg, weights, indices)
        self.criterions = criterions or [CX_loss] * (len(weights))
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterions[i](x_vgg[i], y_vgg[i].detach())
        
        loss = loss[0] if loss.dim() == 1 else loss
        return loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss() # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from 
    # https://arxiv.org/abs/1807.00734        
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1) # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from 
    # https://arxiv.org/abs/1807.00734    
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())
        
        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real


def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    loss_dic = {}

    pixel_loss = ContentLoss()
    # 支持通过 opt.use_charbonnier 将 pixel 损失中的 MSE 替换为 Charbonnier（更鲁棒）
    base_pix_loss = CharbonnierLoss() if getattr(opt, 'use_charbonnier', False) else nn.MSELoss()
    pixel_loss.initialize(MultipleLoss([base_pix_loss, GradientLoss()], [0.2, 0.4]))

    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss

    if opt.lambda_gan > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic
