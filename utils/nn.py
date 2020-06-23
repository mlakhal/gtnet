from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

def count_params(model):
  '''Counts the number of parameters in a model.

  Parameters:
    model (PyTorch model): model definition
  '''
  total_params = sum(p.numel() for p in model.parameters())
  train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  print('Total params: %s' % total_params)
  print('Trainable params: %s' % train_params)
  print('Non-trainable params: %s' % (total_params - train_params))
