'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch.utils.data import Dataset

import warnings
warnings.simplefilter("ignore", UserWarning)

from utils.dataset import get_mesh_flow, to_one_hot, get_ntu_depth
from utils.video import read_frames, read_mask
from utils.string import ntu_replace_camera_id
from utils.os import load_json

class NTU(Dataset):
  def __init__(self,
               dict_path,
               width=224,
               height=224,
               timesteps=8,
               pix_dir='',
               mask_dir='',
               flow_dir='',
               dep_dir='',
               seg_dir='',
               split='train'):

    self.split = split
    self.dict_path = dict_path
    self.pix_dir = pix_dir
    self.flow_dir = flow_dir
    self.dep_dir = dep_dir
    self.seg_dir = seg_dir
    self.mask_dir = mask_dir
    self.width = width
    self.height = height
    self.timesteps = timesteps

    self.dataset = load_json(self.dict_path)

  def __len__(self):
    return len(self.dataset[self.split]['cam1'])

  def _get_indexes(self, video_length):
    if self.split in ['test']:
      ranges = range(video_length - self.timesteps)
      start_index = ranges[len(ranges) // 2]
    else:
      assert video_length >= self.timesteps
      if video_length == self.timesteps:
        start_index = 0
      else:
        start_index = np.random.choice(range(video_length - self.timesteps), 1)[0]
    return start_index

  def _get_data(self, index):
    vid_fn_cam1 = list(self.dataset[self.split]['cam1'].keys())[index]
    vid_fn_cam2 = ntu_replace_camera_id(vid_fn_cam1, '2')
    vid_fn_cam3 = ntu_replace_camera_id(vid_fn_cam1, '3')

    len_cam1 = self.dataset[self.split]['cam1'][vid_fn_cam1]
    len_cam2 = self.dataset[self.split]['cam2'][vid_fn_cam2]
    len_cam3 = self.dataset[self.split]['cam3'][vid_fn_cam3]

    video_length = min(len_cam1, len_cam2, len_cam3)

    return vid_fn_cam1, vid_fn_cam2, vid_fn_cam3, video_length

  def __getitem__(self, index):
    vid_fn_cam1, vid_fn_cam2, vid_fn_cam3, video_length = self._get_data(index)

    start_index = self._get_indexes(video_length)
    frame_indexes = list(range(start_index, start_index+self.timesteps))

    org_dim = 128
    central = True if self.split == 'test' else False

    if self.split == 'train':
      h = np.random.choice(range(org_dim - self.height), 1)[0]
      w = np.random.choice(range(org_dim - self.width), 1)[0]
    else:
      h = org_dim//2 - (self.height//2)
      w = org_dim//2 - (self.width//2)

    kwargs = {'height': org_dim, 'width': org_dim,
              'crop_height': self.height,
              'crop_width': self.width,
              'normalize': True, 'central': central,
              'h': h, 'w': w}

    frames_cam1 = read_frames(vid_fn_cam1, frame_indexes, **kwargs)
    frames_cam2 = read_frames(vid_fn_cam2, frame_indexes, **kwargs)
    frames_cam3 = read_frames(vid_fn_cam3, frame_indexes, **kwargs)

    batch_dict = {'cam1': frames_cam1,
                  'cam2': frames_cam2,
                  'cam3': frames_cam3}

    depth_fold_cam1 = os.path.join(self.dep_dir,
      vid_fn_cam1.split('/')[-1].split('_rgb.avi')[0])
    depth_fold_cam2 = os.path.join(self.dep_dir,
      vid_fn_cam2.split('/')[-1].split('_rgb.avi')[0])
    depth_fold_cam3 = os.path.join(self.dep_dir,
      vid_fn_cam3.split('/')[-1].split('_rgb.avi')[0])

    dep1 = get_ntu_depth(depth_fold_cam1, np.array(frame_indexes) + 1, **kwargs)[0][None, :]
    dep2 = get_ntu_depth(depth_fold_cam2, np.array(frame_indexes) + 1, **kwargs)[0][None, :]
    dep3 = get_ntu_depth(depth_fold_cam3, np.array(frame_indexes) + 1, **kwargs)[0][None, :]

    batch_dict['depth1'] = dep1
    batch_dict['depth2'] = dep2
    batch_dict['depth3'] = dep3

    fn1 = vid_fn_cam1.split('/')[-1]
    fn2 = vid_fn_cam2.split('/')[-1]
    fn3 = vid_fn_cam3.split('/')[-1]

    flow_fold_cam1 = os.path.join(self.flow_dir, fn1)
    flow_fold_cam2 = os.path.join(self.flow_dir, fn2)
    flow_fold_cam3 = os.path.join(self.flow_dir, fn3)

    flow_cam1 = get_mesh_flow(flow_fold_cam1, frame_indexes, **kwargs)
    flow_cam2 = get_mesh_flow(flow_fold_cam2, frame_indexes, **kwargs)
    flow_cam3 = get_mesh_flow(flow_fold_cam3, frame_indexes, **kwargs)

    batch_dict['flow1'] = flow_cam1
    batch_dict['flow2'] = flow_cam2
    batch_dict['flow3'] = flow_cam3

    fn = vid_fn_cam1.split('/')[-1].split('_rgb')[0]
    seg_root = os.path.join(self.seg_dir,
      fn.split('C00')[0] + fn.split('C00')[-1][1:])
    s1_fn = seg_root + '_v1.npy'
    s2_fn = seg_root + '_v2.npy'
    s3_fn = seg_root + '_v3.npy'

    s1 = to_one_hot(s1_fn, frame_indexes, **kwargs)
    s2 = to_one_hot(s2_fn, frame_indexes, **kwargs)
    s3 = to_one_hot(s3_fn, frame_indexes, **kwargs)

    batch_dict['seg1'] = s1
    batch_dict['seg2'] = s2
    batch_dict['seg3'] = s3

    fn = vid_fn_cam1.split('/')[-1].split('.')[0]
    pix_root = os.path.join(self.pix_dir,
      fn.split('C00')[0] + fn.split('C00')[-1][1:])
    pix_root = pix_root.split('_rgb')[0]
    v12_fn = pix_root + '_v12.avi'
    v13_fn = pix_root + '_v13.avi'
    v21_fn = pix_root + '_v21.avi'
    v23_fn = pix_root + '_v23.avi'
    v31_fn = pix_root + '_v31.avi'
    v32_fn = pix_root + '_v32.avi'

    v12_pix = read_frames(v12_fn, frame_indexes, **kwargs)
    v13_pix = read_frames(v13_fn, frame_indexes, **kwargs)
    v21_pix = read_frames(v21_fn, frame_indexes, **kwargs)
    v23_pix = read_frames(v23_fn, frame_indexes, **kwargs)
    v31_pix = read_frames(v31_fn, frame_indexes, **kwargs)
    v32_pix = read_frames(v32_fn, frame_indexes, **kwargs)

    batch_dict['v12_pix'] = np.array(v12_pix)
    batch_dict['v13_pix'] = np.array(v13_pix)
    batch_dict['v21_pix'] = np.array(v21_pix)
    batch_dict['v23_pix'] = np.array(v23_pix)
    batch_dict['v31_pix'] = np.array(v31_pix)
    batch_dict['v32_pix'] = np.array(v32_pix)

    v1_fn = os.path.join(self.mask_dir, vid_fn_cam1.split('/')[-1] + '_track-0_faces_output.npy')
    v2_fn = os.path.join(self.mask_dir, vid_fn_cam2.split('/')[-1] + '_track-0_faces_output.npy')
    v3_fn = os.path.join(self.mask_dir, vid_fn_cam3.split('/')[-1] + '_track-0_faces_output.npy')

    mask_cam1 = read_mask(v1_fn, frame_indexes, **kwargs)
    mask_cam2 = read_mask(v2_fn, frame_indexes, **kwargs)
    mask_cam3 = read_mask(v3_fn, frame_indexes, **kwargs)

    batch_dict['mask1'] = mask_cam1
    batch_dict['mask2'] = mask_cam2
    batch_dict['mask3'] = mask_cam3

    return batch_dict
