'''Novel-View Human Action Synthesis

Implementation of the perceptual loss defined in:
    View-LSTM: Novel-view video synthesis through view decomposition. In Proc. ICCV, 2019

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models.i3d import InceptionI3d
from utils.nn import count_params

I3D_path = './data/i3d/rgb_imagenet.pt'

class PerceptualLoss(nn.Module):

  def __init__(self, temporal=False, weights=[], device='cpu', i3d_path=I3D_path,
               use_logits=False, finetune=False):
    super(PerceptualLoss, self).__init__()

    self.temporal = temporal
    self.use_logits = use_logits
    self.mse_loss = torch.nn.MSELoss()

    if not weights:
      weights = [1., 1., 1., 1.]
    else:
      assert len(weights) == 4
    self.weights = weights

    if temporal:
      action = True if use_logits else False
      model = InceptionI3d(400, in_channels=3, action=action)

      model.load_state_dict(torch.load(i3d_path))
      model.set_requires_grad()
      if finetune:
        model.replace_logits(60)
      self.model = model.to(device)
    else:
      self.model = Vgg16(requires_grad=False).to(device)

    count_params(self.model)

  def forward(self, x_pred, x_true):
    features_pred = self.model(x_pred)
    features_true = self.model(x_true)

    loss = self.weights[0] * self.mse_loss(features_pred.h1, features_true.h1)
    loss += self.weights[1] * self.mse_loss(features_pred.h2, features_true.h2)
    loss += self.weights[2] * self.mse_loss(features_pred.h3, features_true.h3)
    loss += self.weights[3] * self.mse_loss(features_pred.h4, features_true.h4)

    if self.use_logits:
      loss += self.mse_loss(self.model(x_pred, logits=True),
                            self.model(x_true, logits=True))

    return loss
