'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

from .gtnet import GTNet

__models__ = {
'GTNet': GTNet,
}

def _check_model(model_name):
  if not model_name in __models__.keys():
    raise KeyError('model %s not supported!' % model_name)

def load_model(model_name, **kwargs):
  _check_model(model_name)
  pp = pprint.PrettyPrinter(indent=4)
  return __models__[model_name](**kwargs)
