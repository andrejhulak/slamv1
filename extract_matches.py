import cv2, os
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def generate_match(f1, f2):
  bf = cv2.BFMatcher(cv2.NORM_HAMMING)
  matches = bf.knnMatch(f1.des, f2.des, k=2)

  ret = []
  idx1, idx2 = [], []
  idx1s, idx2s = set(), set()

  for m,n in matches:
    if m.distance < 0.75*n.distance:
      p1 = f1.kpus[m.queryIdx]
      p2 = f2.kpus[m.trainIdx]

      if m.distance < 32 or True:
        if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
          idx1.append(m.queryIdx)
          idx2.append(m.trainIdx)
          idx1s.add(m.queryIdx)
          idx2s.add(m.trainIdx)
          ret.append((p1, p2))
    
  assert(len(set(idx1)) == len(idx1))
  assert(len(set(idx2)) == len(idx2))

  ret = np.array(ret)
  idx1 = np.array(idx1)
  idx2 = np.array(idx2)

  E, mask = cv2.findEssentialMat(ret[:, 0], ret[:, 1], f1.K)
  _, R, t, _ = cv2.recoverPose(E, ret[:, 0], ret[:, 1], f1.K)

  return idx1[mask[:, 0] == 1], idx2[mask[:, 0] == 1], poseRt(R, t.squeeze())

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret