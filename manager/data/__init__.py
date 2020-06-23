'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import

from torch.utils import data
from .datasets import load_dataset

def get_generator(dict_path='',
                  dataset_name='NTU',
                  **kwargs):
  if dataset_name == 'NTU':
    args = [dict_path]
    dataset_kwargs = kwargs['dataset']['ntu']
    loader_kwargs = kwargs['loader']
  else:
    args = []
    dataset_kwargs = {}
    loader_kwargs = {}

  dataset = load_dataset(dataset_name, *args, **dataset_kwargs)
  data_generator = data.DataLoader(dataset, **loader_kwargs)

  return data_generator
