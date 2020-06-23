'''Novel-View Human Action Synthesis

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def ntu_replace_camera_id(video_path, camera_id='1'):
  '''
  Replace the camera_id from a video path of NTU RGB+D dataset

  Parameters:
    video_path (str): valid filename
    camera_id (str): the camera_id to replace with

  Returns:
      (str): new path with camera_id
  '''
  assert type(video_path) == str and type(camera_id) == str
  assert camera_id.isdigit()
  assert int(camera_id) in [1, 2, 3]
  if not os.path.isfile(video_path):
    raise FileNotFoundError

  return video_path.split('C00')[0] + 'C00%s' % camera_id + video_path.split('C00')[-1][1:]
