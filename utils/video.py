'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import lintel

def warp_function(x, flow, padding_mode='zeros'):
  """Warp an image or feature map with optical flow
  Refs: https://github.com/hellock/cvbase/
  Args:
      x (Tensor): size (n, c, h, w)
      flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)
      padding_mode (str): 'zeros' or 'border'

  Returns:
      Tensor: warped image or feature map
  """
  assert x.size()[-2:] == flow.size()[-2:]
  n, _, h, w = x.size()
  x_ = torch.arange(w).view(1, -1).expand(h, -1)
  y_ = torch.arange(h).view(-1, 1).expand(-1, w)
  grid = torch.stack([x_, y_], dim=0).float().cuda()
  grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
  grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
  grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
  grid += 2 * flow
  grid = grid.permute(0, 2, 3, 1)
  return F.grid_sample(x, grid, padding_mode=padding_mode)

def read_mask(mask_filename,
              frame_nums,
              height=256,
              width=256,
              crop_height=224,
              crop_width=224,
              normalize=False,
              central=False,
              h=-1, w=-1):
  p = np.load(mask_filename)[frame_nums]
  idxs = np.argwhere(p > 0.)
  mask_org = np.zeros(p.shape)
  mask_org[idxs[:,0], idxs[:,1], idxs[:,2]] = 1.
  mask = []
  for m in mask_org:
    m = cv2.resize(m, (height, width), interpolation=cv2.INTER_NEAREST)
    mask.append(m)
  mask = np.array(mask)

  if central:
    h = height//2 - (crop_height//2)
    w = width//2 - (crop_width//2)
  else:
    if h == -1 and w == -1:
      h = np.random.choice(range(height - crop_height), 1)[0]
      w = np.random.choice(range(width - crop_width), 1)[0]

  mask = mask[:, h:(crop_height + h), w:(crop_width + w)]

  return mask

def read_frames(video_filename,
                frame_nums,
                height=256,
                width=256,
                crop_height=224,
                crop_width=224,
                normalize=False,
                central=False,
                h=-1, w=-1,
                use_crop=True):
  with open(video_filename, 'rb') as f:
    encoded_video = f.read()
    decoded_frames = lintel.loadvid_frame_nums(encoded_video,
                                               frame_nums=frame_nums,
                                               width=width,
                                               height=height)
    decoded_frames = np.frombuffer(decoded_frames, dtype=np.uint8)
    decoded_frames = np.reshape(
        decoded_frames,
        newshape=(len(frame_nums), height, width, 3))
    decoded_frames = decoded_frames.transpose([3, 0, 1, 2])
    decoded_frames = decoded_frames.astype(np.float32)

    if normalize:
      decoded_frames = (decoded_frames / 255. - 0.5) * 2

    if use_crop:
      if central:
        h = height//2 - (crop_height//2)
        w = width//2 - (crop_width//2)
      else:
        if h == -1 and w == -1:
          h = np.random.choice(range(height - crop_height), 1)[0]
          w = np.random.choice(range(width - crop_width), 1)[0]

      decoded_frames = decoded_frames[:, :, h:(crop_height + h), w:(crop_width + w)]

    return decoded_frames
