from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
from collections import OrderedDict
from .networks import get_scheduler

class BaseModel(object):
  def __init__(self, is_train=True):
    super(BaseModel, self).__init__()

    self.is_train = is_train
    self.loss_names = []

  def set_input(self, input):
    raise NotImplementedError

  def forward(self):
    raise NotImplementedError

  def setup(self, lr_policy, optim_kwargs):
    raise NotImplementedError

  def eval(self):
    for name in self.model_names:
      if isinstance(name, str):
        net = getattr(self, 'net' + name)
        net.eval()

  def test(self):
    with torch.no_grad():
      self.forward()

  def optimize_parameters(self):
    raise NotImplementedError

  def update_learning_rate(self):
    raise NotImplementedError

  def get_current_visuals(self):
    visual_ret = OrderedDict()
    for name in self.visual_names:
      if isinstance(name, str):
        visual_ret[name] = getattr(self, name)
    return visual_ret

  def get_current_losses(self):
    errors_ret = OrderedDict()
    for name in self.loss_names:
      if isinstance(name, str):
        errors_ret[name] = float(getattr(self, 'loss_' + name))
    return errors_ret

  def save_networks(self, save_path, epoch):
    raise NotImplementedError

  def load_networks(self, epoch):
    raise NotImplementedError

  def print_networks(self, verbose):
    raise NotImplementedError

  def set_requires_grad(self, nets, requires_grad=False):
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad
