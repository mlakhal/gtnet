'''Novel-View Human Action Synthesis

'''
import numpy as np

from collection import Counter, defaultdict

sym_part = [5,6,7,8,1,2,3,4,9,10]

with open('data/FAUST_sym_idxs.txt') as f:
  sym_idxs = f.readlines()
sym_idxs = [int(s) for s in sym_idxs]

with open('data/labels.txt') as f:
  labels = f.readlines()
labels = [int(s) for s in labels]

# pre-computed pairwise distance between faces
F = np.load('data/face_pairwise_dist.npy')

faces_to_label = []
for face in faces:
  v1, v2, v3 = face
  cnt = Counter([labels[v1],labels[v2],labels[v3]])
  label = cnt.most_common()[0][0]
  faces_to_label.append(label)
faces_to_label = np.array(faces_to_label)

def step_I(x_i, F_i, T):
  '''
  Step I. Face to pixel hashmap (Alg. 2 supp)

  Parameters:
    x_i (numpy.array): video stream of view i
    F_i (numpy.array): face-map of view i
    T (int): number of timesteps

  Returns:
      (dict)
  '''

  face_to_pixs = defaultdict(list)
  for t in range(T):
    idxs_i = np.argwhere(F_i[t] >= 0)
    for (x_f, y_f) in idxs_i:
      f_i = F_i[t][x_f, y_f]
      face_to_pixs[f_i].append(list(x_i[t][x_f, y_f]))

  F_dict = {}
  for k, v in face_to_pixs.items():
    if len(v) == 1:
      v = v[0]
    else:
      v = list(np.median(v, axis=0))
    F_dict[k] = v

  return F_dict

def step_II(F_jt, F_dict, n=50, size=(128,128,3)):
  '''
  Step II. Nearest Neighbor Texture Transfer (Alg. 3 supp)
  Complexity O(WxH) i.e. W: frame-width, H: frame-height

  Parameters:
    F_jt (numpy.array): face-map of view j at timestep t
    F_dict (dict): Step I results
    n (int), default 50: nearest neighbor value
    size (tuple): default dimension

  Returns:
      (numpy.array)
  '''

  # \mathcal{T}_{i \to j}
  T_ij = np.zeros(size)
  # loop over the visible faces
  idxs_j = np.argwhere(F_jt >= 0)
  label_to_pix = defaultdict(list)
  O_xy = []

  for (u_x, u_y) in idxs_j:
    f_j = F_jt[u_x,u_y]

    if not f_j in F_dict.keys():
      k_cnt = 1
      cnt = True
      f_q = F[f_j]

      while cnt:
        q = f_q[k_cnt]
        if q in F_dict.keys():
          T_ij[u_x, u_y] = F_dict[q]
          k_lab = faces_to_label[f_j]
          label_to_pix[k_lab].append([F_dict[q], [u_x,u_y,0]])
          cnt = False
        else:
          k_cnt += 1
          if k_cnt == n + 1:
            O_xy.append([u_x,u_y])
            cnt = False
    else:
      T_ij[u_x, u_y] = F_dict[f_j]
      k_lab = faces_to_label[f_j]
      label_to_pix[k_lab].append([F_dict[q], [u_x,u_y,0]])

  return T_ij, O_xy, label_to_pix

def step_III(x_i, F_i, F_jt, n=50, size=(128,128,3)):
  '''
  Step III. Symmetric Texture Transfer (Alg. 1)

  Parameters:
    x_i (numpy.array): video stream of view i
    F_i (numpy.array): face-map of view i
    F_jt (numpy.array): face-map of view j at timestep t
    n (int), default 50: nearest neighbor value
    size (tuple), default (128,128,3): texture map dimension

  Returns:
      (numpy.array)
  '''

  T = x_i.shape[0]
  F_dict = step_I(x_i, F_i, T)
  T_s_ij, O_xy, label_to_pix = step_II(F_jt, F_dict, n=n, size=size)

  for (u_x, u_y) in O_xy:
    face = F_jt[u_x, u_y]

    label = faces_to_label[face]
    if not label in label_to_pix.keys():
      label = sym_part[label-1]
      if not label in label_to_pix.keys():
        # special-value (i.e non-visible)
        T_s_ij[u_x, u_y] = [102, 163, 255]
        continue

    vs = np.array(label_to_pix[label])
    v_pix = np.array(vs[:,0])
    v_xy = np.array(vs[:,1])
    v_xy = v_xy[:,:2]
    v_q = np.array([[u_x, u_y]])

    # pairwise distances
    P = np.add.outer(np.sum(v_xy**2, axis=1), np.sum(v_q**2, axis=1))
    N = np.dot(v_xy, v_q.T)
    dists = np.sqrt(P - 2*N).reshape(-1)

    T_s_ij[u_x,u_y] = v_pix[np.argmin(dists)]

  # \mathcal{T}^s_{i \to j}
  return T_s_ij
