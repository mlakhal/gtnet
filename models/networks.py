'''Novel-View Human Action Synthesis

'''

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

from utils.video import warp_function # \mathcal{W} (see Eq. 4)

NGF = 64
NDF = 64
n_blocks = 6
norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
use_bias = norm_layer.func == nn.InstanceNorm3d
n_downsampling = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_scheduler(optimizer):
  scheduler = lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
  return scheduler

def define_G(timesteps=16,
             zeta=.1,
             split='train',
             model_name=''):
  if model_name == 'GTNet':
    net = GTNet(timesteps=timesteps, split=split)
  else:
    raise NotImplementedError

  return net.to(device)

def define_D(input_nc):
  norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
  net = Discriminator(input_nc, NDF, n_layers=3, norm_layer=norm_layer)
  return net.to(device)

class GANLoss(nn.Module):
  def __init__(self, target_real_label=1.0, target_fake_label=0.0):
    super(GANLoss, self).__init__()
    self.register_buffer('real_label', torch.tensor(target_real_label))
    self.register_buffer('fake_label', torch.tensor(target_fake_label))
    self.loss = nn.BCELoss()

  def get_target_tensor(self, input, target_is_real):
    if target_is_real:
      target_tensor = self.real_label
    else:
      target_tensor = self.fake_label
    return target_tensor.expand_as(input)

  def __call__(self, input, target_is_real):
    target_tensor = self.get_target_tensor(input, target_is_real)
    return self.loss(input, target_tensor)

class Resnet3DBlock(nn.Module):
  def __init__(self, dim):
    super(Resnet3DBlock, self).__init__()
    self.conv_block = self.build_conv_block(dim)

  def build_conv_block(self, dim):
    conv_block = []
    conv_block += [nn.ConstantPad3d(1,0)]
    conv_block += [nn.Conv3d(dim,
                             dim,
                             kernel_size=3,
                             padding=0,
                             bias=use_bias_3d),
                   norm_layer_3d(dim),
                   nn.ReLU(True)]

    conv_block += [nn.ConstantPad3d(1,0)]
    conv_block += [nn.Conv3d(dim,
                             dim,
                             kernel_size=3,
                             padding=0,
                             bias=use_bias_3d),
                   norm_layer_3d(dim)]

    return nn.Sequential(*conv_block)

  def forward(self, x):
    out = x + self.conv_block(x)
    return out

##############################################
##        Implementation of GTNet           ##
##############################################

class GTNet(nn.Module):
  def __init__(self,
               timesteps=16,
               zeta=.1,
               split='train'):
    super(GTNet, self).__init__()
    self.split = split
    self.timesteps = timesteps # T
    self.zeta = zeta # \zeta

    self.E_x = self._encoder() # \mathcal{E}_{\theta_x}
    self.E_d = self._encoder() # \mathcal{E}_{\theta_d}
    self.E_S = self._encoder() # \mathcal{E}_{\theta_{\mathcal{S}}}
    self.E_T = self._encoder() # \mathcal{E}_{\theta_{\mathcal{T}}}

    # \Psi_{i \to j} = \Psi^{\texttt{conv}}_{i \to j}
    self.psi = nn.Conv3d(1024, 256, kernel_size=3, stride=1, padding=1)

    self.D_f = self._decoder() # \mathcal{D}_{\theta_f}
    self.D_b = self._decoder() # \mathcal{D}_{\theta_b}

    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
      elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _encoder(self, input_nc=3):
    model = [nn.ConstantPad3d(3,0),
             nn.Conv3d(input_nc,
                       NGF,
                       kernel_size=7,
                       padding=0,
                       bias=use_bias),
             norm_layer(NGF),
             nn.ReLU(True)]

    for i in range(n_downsampling):
      mult = 2**i
      model += [nn.Conv3d(NGF * mult,
                          NGF * mult * 2,
                          kernel_size=3,
                          stride=2,
                          padding=1,
                          bias=use_bias),
                norm_layer(NGF * mult * 2),
                nn.ReLU(True)]

    mult = 2**n_downsampling
    for i in range(n_blocks):
      model += [Resnet3DBlock(NGF * mult)]

    return nn.Sequential(*model)

  def _decoder(self, output_nc=3):
    model = []
    for i in range(n_downsampling):
      mult = 2**(n_downsampling - i)
      model += [nn.ConvTranspose3d(NGF * mult,
                                   int(NGF * mult / 2),
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1,
                                   bias=use_bias_3d),
                norm_layer(int(NGF * mult / 2)),
                nn.ReLU(True)]
    model += [nn.ConstantPad3d(3,0)]
    model += [nn.Conv3d(NGF, output_nc, kernel_size=7, padding=0)]
    model += [nn.Tanh()]

    return nn.Sequential(*model)

  def forward(self, x_i, d_j, T_ij, S_j, O_j):
    x_i = self.E_x(x_i)
    d_j = self.E_d(d_j)
    S_j = self.E_S(S_j)
    T_ij = self.E_T(T_ij)

    input = torch.cat((x_i, d_j, S_j, T_ij), dim=1)
    epsilon_j = self.psi(input) # Eq. 2

    x_f = self.D_f(epsilon_j)
    x_b = self.D_b(epsilon_j)

    if self.zeta: # Eq. 4
      x_tilda = x_f
      # we don't backprop here (see Sec. 5.1)
      x_j_prev = x_tilda[:,:,:-1,:,:].detach()
      O_j = O_j[:,:,1:,:,:]
      x_j_warp = []
      # t \in [2..T]
      for t in range(self.timesteps - 1):
        x_jt = x_j_prev[:,:,t,:,:]
        O_jt = O_j[:,:,t,:,:]
        x_jt_warp = warp_function(x_jt, O_jt)
        x_j_warp.append(x_jt_warp)

      x_j_warp = torch.stack(x_j_warp).permute(1,2,0,3,4)
      x_f_t1 = x_f[:,:,0,:,:].unsqueeze(2) # Eq. 4 if t = 1
      x_f_tk = x_tilda[:,:,1:,:,:] + self.zeta * x_j_warp # Eq. 4 if t \in [2..T]
      x_f = torch.cat([x_f_t1, x_f_tk], dim=2)

    return x_f, x_b

class Discriminator(nn.Module):
  def __init__(self,
               input_nc,
               ndf=64,
               n_layers=3,
               norm_layer=nn.BatchNorm2d,
               use_sigmoid=True):
    super(Discriminator, self).__init__()
    if type(norm_layer) == functools.partial:
      use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
      use_bias = norm_layer == nn.InstanceNorm2d

    kw = 4
    padw = 1
    sequence = [
      nn.Conv2d(input_nc,
                ndf,
                kernel_size=kw,
                stride=2,
                padding=padw),
      nn.LeakyReLU(0.2, True)
    ]

    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):
      nf_mult_prev = nf_mult
      nf_mult = min(2**n, 8)
      sequence += [
        nn.Conv2d(ndf * nf_mult_prev,
                  ndf * nf_mult,
                  kernel_size=kw,
                  stride=2,
                  padding=padw,
                  bias=use_bias),
        norm_layer(ndf * nf_mult),
        nn.LeakyReLU(0.2, True)
      ]

    nf_mult_prev = nf_mult
    nf_mult = min(2**n_layers, 8)
    sequence += [
      nn.Conv2d(ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias),
      norm_layer(ndf * nf_mult),
      nn.LeakyReLU(0.2, True)
    ]

    sequence += [nn.Conv2d(ndf * nf_mult, 1,
                           kernel_size=kw,
                           stride=1,
                           padding=padw)]

    if use_sigmoid:
      sequence += [nn.Sigmoid()]

    self.model = nn.Sequential(*sequence)

  def forward(self, input):
    b, c, t, w, h = input.shape
    input = input.transpose(2,1).reshape(b * t, c, w, h)
    return self.model(input)
