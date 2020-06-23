import distutils.util
import argparse

def argument_parser():
  parser = argparse.ArgumentParser()

  # ************************************************************
  # Datasets (NTU)
  # ************************************************************
  parser.add_argument('--zeta', type=float, default=.1)
  parser.add_argument('--load_D', type=str, default='')
  parser.add_argument('--resume', type=str, default='')
  parser.add_argument('--model_name', type=str, default='')
  parser.add_argument('--split', type=str, default='train')
  parser.add_argument('--load_path', type=str, default='')
  parser.add_argument('--save_path', type=str, default='')
  parser.add_argument('--dataset_name', type=str, default='')
  parser.add_argument('--dataset_dict_path', type=str, default='')
  parser.add_argument('--num_workers', default=4, type=int)
  parser.add_argument('--image_height', type=int, default=224)
  parser.add_argument('--image_width', type=int, default=224)
  parser.add_argument('--timesteps', type=int, default=15)
  parser.add_argument('--max_epoch', default=60, type=int)
  parser.add_argument('--start_epoch', default=0, type=int)
  parser.add_argument('--train_batch_size', default=32, type=int)
  parser.add_argument('--pix_dir', type=str, default='')
  parser.add_argument('--flow_dir', type=str, default='')
  parser.add_argument('--dep_dir', type=str, default='')
  parser.add_argument('--seg_dir', type=str, default='')
  parser.add_argument('--mask_dir', type=str, default='')
  # ************************************************************
  # Miscs
  # ************************************************************
  parser.add_argument('--print_every', type=int, default=100)
  parser.add_argument('--save_every', type=int, default=5)
  parser.add_argument('--log_every', type=int, default=100)
  parser.add_argument('--logs_dir', type=str, default='log')

  return parser

def models_kwargs(parsed_args, split='train'):
  dropout_keep_prob=1.0
  is_train = True if split == 'train' else False
  if split == 'train':
    dropout_keep_prob=.5
    batch_size = parsed_args.train_batch_size

  arch_dict = {
      'is_train': is_train,
      'timesteps': parsed_args.timesteps,
      'zeta': parsed_args.zeta,
    }
  return arch_dict

def split_kwargs(parsed_args, split='train'):
  shuffle = True if split == 'train' else False
  if split == 'train':
    batch_size = parsed_args.train_batch_size

  return {'dataset':{
      'ntu':{'width': parsed_args.image_width,
             'height': parsed_args.image_height,
             'timesteps': parsed_args.timesteps,
             'pix_dir': parsed_args.pix_dir,
             'flow_dir': parsed_args.flow_dir,
             'dep_dir': parsed_args.dep_dir,
             'seg_dir': parsed_args.seg_dir,
             'mask_dir': parsed_args.mask_dir,
             'split': split,
      },
    },
    'loader':{
      'batch_size': batch_size,
      'shuffle': shuffle,
      'num_workers': parsed_args.num_workers,
    }
  }
