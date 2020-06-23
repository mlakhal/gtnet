'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import

from .ntu import NTU

def load_dataset(dataset_name, *args, **kwargs):
  if dataset_name == 'NTU':
    dataset = NTU(*args, **kwargs)
    return dataset
  else:
   raise KeyError('dataset %s not supported' % dataset_name)
