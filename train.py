'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import time
import numpy as np

from models import load_model
from manager.data import get_generator
from utils.dataset import AverageMeter
from utils.os import makedir
from utils.logger import Logger
from utils.nn import count_params
from args import argument_parser, split_kwargs, models_kwargs

parser = argument_parser()
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, dataset):
  batch_time = AverageMeter()
  losses = AverageMeter()

  len_dataset = len(dataset)
  end_time = time.time()

  logger = Logger(args.logs_dir)
  itr = 0

  for epoch in range(args.start_epoch, args.max_epoch + 1):

    for batch in dataset:
      x_i = torch.cat((batch['cam1'],
                       batch['cam2'],
                       batch['cam3']),
                       dim=0).to(device)

      cam1j = np.random.choice([2,3])
      cam2j = np.random.choice([1,3])
      cam3j = np.random.choice([1,2])

      T_ij = torch.cat((batch['v1%s_pix' % cam1j],
                        batch['v2%s_pix' % cam2j],
                        batch['v3%s_pix' % cam3j]),
                        dim=0).type(torch.FloatTensor).to(device)

      x_j = torch.cat((batch['cam%s' % cam1j],
                       batch['cam%s' % cam2j],
                       batch['cam%s' % cam3j]),
                       dim=0).to(device)

      m_j = torch.cat((batch['mask%s' % cam1j],
                       batch['mask%s' % cam2j],
                       batch['mask%s' % cam3j]),
                       dim=0).type(torch.FloatTensor).to(device)
      m_j = m_j.unsqueeze(1)

      S_j = torch.cat((batch['seg%s' % cam1j],
                       batch['seg%s' % cam2j],
                       batch['seg%s' % cam3j]),
                       dim=0).to(device).permute(0,4,1,2,3)

      O_j = torch.cat((batch['flow%s' % cam1j],
                       batch['flow%s' % cam2j],
                       batch['flow%s' % cam3j]),
                       dim=0).type(torch.FloatTensor).to(device)

      d_j = torch.cat((batch['depth%s' % cam1j],
                       batch['depth%s' % cam2j],
                       batch['depth%s' % cam3j]),
                       dim=0).type(torch.FloatTensor).to(device)

      input = {'x_i': x_i,
              'x_j': x_j,
              'T_ij': T_ij,
              'S_j': S_j,
              'O_j': O_j,
              'd_j': d_j,
              'm_j': m_j}

      model.set_input(input)
      model.optimize_parameters(epoch)

      batch_time.update(time.time() - end_time)
      end_time = time.time()
      losses.update(model.get_loss_value(), I_i.size(0))
      if itr % args.print_every == 0:
        model.print_networks(epoch, itr, len_dataset, batch_time, losses)

      if itr % args.log_every == 0:
        logger.scalar_summary('recons_loss', model.get_loss_value(), itr)

      itr += 1

    if epoch % args.save_every == 0:
      model.save_networks(args.save_path, epoch)

def main():
  train_kwargs = split_kwargs(args, split='train')
  training_generator = get_generator(dict_path=args.dataset_dict_path,
                                     dataset_name=args.dataset_name,
                                     **train_kwargs)

  model = load_model(args.model_name, **models_kwargs(args, split='train'))
  count_params(model)

  if args.resume:
    model.load_networks(args.resume, load_D=args.load_D, with_opt=True)

  makedir(args.save_path)
  train(model, training_generator)

if __name__ == '__main__':
  main()
