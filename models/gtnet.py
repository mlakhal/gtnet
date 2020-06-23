'''Novel-View Human Action Synthesis

'''

import os
import torch
import torch.nn as nn

from .base_model import BaseModel
from .networks import define_G, define_D, GANLoss
from utils.nn import save_checkpoint
from losses.perceptual import PerceptualLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GTNet(BaseModel):
  def __init__(self, timesteps=16, zeta=.1, is_train=True):
    super(GTNet, self).__init__()

    self.is_train = is_train
    split = 'train' if is_train else 'test'
    self.timesteps = timesteps

    self.G_ij = define_G(timesteps=timesteps, split=split, zeta=zeta, model_name='GTNet')

    if self.is_train:
      self.netD = define_D(6)
      self.criterionGAN = GANLoss().to(device)

      self.crit_cnt = torch.nn.SmoothL1Loss(reduction='sum')
      self.crit_per = PerceptualLoss(temporal=True, device=device)

      self.optimizers = []
      self.optimizer_G = torch.optim.Adam(self.G_ij.parameters(), lr=2e-5, betas=(0.5, 0.999))
      self.optimizers.append(self.optimizer_G)
      self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=2e-5, betas=(0.5, 0.999))
      self.optimizers.append(self.optimizer_D)

  def set_input(self, input):
    self.x_i = input['x_i']
    self.T_ij = input['T_ij']
    self.m_j = input['m_j']
    self.x_j = input['x_j']
    self.d_j = input['d_j']
    self.S_j = input['S_j']
    self.O_j = input['O_j']

  def forward(self):
    self.xp_j_f, self.xp_j_b = self.G_ij(self.x_i, self.d_j, self.T_ij, self.S_j, self.O_j)
    self.xp_j = self.xp_j_f * self.m_j + self.xp_j_b * (1 - self.m_j) # Eq. 3

  def backward_D(self):
    fake_ij = torch.cat((self.x_i, self.xp_j), 1)
    pred_fake = self.netD(fake_ij.detach())
    self.loss_D_fake = self.criterionGAN(pred_fake, False)

    real_ij = torch.cat((self.x_i, self.x_j), 1)
    pred_real = self.netD(real_ij)
    self.loss_D_real = self.criterionGAN(pred_real, True)

    self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    self.loss_D.backward()

  def backward_G(self, epoch):
    xp_j_f = self.xp_j_fg * self.m_j
    x_j_f = self.x_j * self.m_j
    xp_j_b = self.xp_j_bg * (1 - self.m_j)
    x_j_b = self.x_j * (1 - self.m_j)

    L_f = self.crit_cnt(x_j_f, x_j_f) / (self.m_j.sum() + 1e-9) # numerical stability
    L_b = self.crit_cnt(x_j_b, x_j_b) / ((1 - self.m_j).sum() + 1e-9)

    L_p = self.crit_per(self.xp_j, self.x_j) # Eq. 6
    self.L_r =  L_f + L_b # Eq. 5

    fake_ij = torch.cat((self.x_i, self.xp_j), 1)
    pred_fake = self.netD(fake_ij)
    L_a = self.criterionGAN(pred_fake, True)

    # total loss
    L = self.L_r + .01 * L_p + .01 * L_a
    L.backward()

  def optimize_parameters(self, epoch):
    self.forward()

    self.set_requires_grad(self.netD, True)
    self.optimizer_D.zero_grad()
    self.backward_D()
    self.optimizer_D.step()
    self.set_requires_grad(self.netD, False)

    self.optimizer_G.zero_grad()
    self.backward_G(epoch)
    self.optimizer_G.step()

  def get_loss_value(self):
    return float(self.L_r.item())

  def print_networks(self, epoch, i, len_dataset, batch_time, losses):
    print('Epoch: [{0}][{1}/{2}]\t'
          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'L_recons {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch,
              (i+1) % len_dataset,
              len_dataset,
              batch_time=batch_time,
              loss=losses))

  def load_networks(self, load_path, load_D='', with_opt=False):
    print('Generator model')
    if os.path.isfile(load_path):
      checkpoint = torch.load(load_path)
      self.G_ij.load_state_dict(checkpoint['state_dict'])
      if with_opt:
        self.optimizer_G.load_state_dict(checkpoint['optimizer'])
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(load_path, checkpoint['epoch']))
      if os.path.isfile(load_D):
        checkpoint = torch.load(load_D)
        self.netD.load_state_dict(checkpoint['state_dict'])
        if with_opt:
          self.optimizer_D.load_state_dict(checkpoint['optimizer'])
    else:
      raise KeyError("=> no checkpoint found at '{}'".format(load_path))

  def save_networks(self, save_path, epoch):
    # Net-G
    state = {
      'epoch': epoch,
      'state_dict': self.G_ij.state_dict(),
      'optimizer' : self.optimizer_G.state_dict(),
    }
    filename = os.path.join(save_path,
      'generator_epoch_%s.pth.tar' % (epoch))
    save_checkpoint(state, filename)

    # Net-D
    state = {
      'epoch': epoch,
      'state_dict': self.netD.state_dict(),
      'optimizer' : self.optimizer_D.state_dict(),
    }
    filename = os.path.join(save_path,
      'discriminator_epoch_%s.pth.tar' % (epoch))
    save_checkpoint(state, filename)
