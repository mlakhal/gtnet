from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def to_one_hot(seg_fn, frame_nums,
               height=256, width=256,
               crop_height=224, crop_width=224,
               normalize=False,
               central=False, h=-1, w=-1, n_dims=11):
  ''' Code adapted from:
  https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/23
  '''
  segs = np.load(seg_fn)
  segs = segs[frame_nums]

  if central:
    h = height//2 - (crop_height//2)
    w = width//2 - (crop_width//2)
  else:
    if h == -1 and w == -1:
      h = np.random.choice(range(height - crop_height), 1)[0]
      w = np.random.choice(range(width - crop_width), 1)[0]

  segs = segs[:, h:(crop_height + h), w:(crop_width + w)]
  y = torch.from_numpy(segs)

  y_tensor = y.data if isinstance(y, Variable) else y
  y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
  n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
  y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
  y_one_hot = y_one_hot.view(*y.shape, -1)

  return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot

def rgb_to_flow(rgb, normalize=False):
  flow = rgb
  flow = flow.astype(np.float)
  flow /= 255.
  flow *= 40.
  flow -= 20.
  if normalize:
    flow /= 20
  return flow

def get_mesh_flow(flow_fn, frame_nums,
                  height=256, width=256,
                  crop_height=224, crop_width=224,
                  central=False, normalize=False,
                  h=-1, w=-1):

  flow_x = read_frames(flow_fn + '_flow_x.mp4', frame_nums,
                       height=height, width=width,
                       crop_height=crop_height,
                       crop_width=crop_width,
                       normalize=False, central=central, h=h, w=w)

  flow_y = read_frames(flow_fn + '_flow_y.mp4', frame_nums,
                       height=height, width=width,
                       crop_height=crop_height,
                       crop_width=crop_width,
                       normalize=False, central=central, h=h, w=w)

  flows = np.concatenate([np.expand_dims(flow_x[0], 0),
                          np.expand_dims(flow_y[0], 0)], axis=0)
  flows = rgb_to_flow(flows, normalize=normalize)

  return flows

def get_ntu_depth(depth_folder, frame_nums,
                  height=256, width=256,
                  crop_height=224, crop_width=224,
                  central=False, normalize=False,
                  h=-1, w=-1):
  depths = []
  for frame_num in frame_nums:
    frame_num_str = 'Depth-'+str(frame_num).zfill(8)+'.png'
    depth_path = os.path.join(depth_folder, frame_num_str)
    depth = cv2.imread(depth_path, -1)
    if type(depth) == type(None):
      print(depth_path)
    depths.append(depth)

  depths = np.array(depths)
  depths = depths.transpose(3, 0, 1, 2)
  if normalize:
    depths = (depths / 255. - 0.5) * 2

    if central:
      h = height//2 - (crop_height//2)
      w = width//2 - (crop_width//2)
    else:
      if h == -1 and w == -1:
        h = np.random.choice(range(height - crop_height), 1)[0]
        w = np.random.choice(range(width - crop_width), 1)[0]

    depths = depths[:, :, h:(crop_height + h), w:(crop_width + w)]

  return depths
