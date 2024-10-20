import cv2
import numpy as np
from extract_matches import *
from frame import denormalize

def get_camera_position(pose):
  R = pose[0:3, 0:3]
  t = pose[0:3, 3]
  camera_position = -np.dot(R.T, t)
  return camera_position

def triangulate(img, frame1, frame2, Rt, idx1, idx2):
  K = frame1.K
  frame2.pose = np.dot(Rt, frame1.pose)
  pts4d = cv2.triangulatePoints(frame1.pose[:3, :], frame2.pose[:3, :], frame1.kps[idx1].T, frame2.kps[idx2].T).T
  pts4d /= pts4d[:, 3:]

  pts3d = []
  rip = 0
  proj_err_treshold = 0.001
  for i,p in enumerate(pts4d):
    pl1 = np.dot(frame1.pose, p)
    pl2 = np.dot(frame2.pose, p)
    if pl1[2] < 0 or pl2[2] < 0:
      continue

    pp1 = np.dot(K, pl1[:3])
    pp2 = np.dot(K, pl2[:3])
    pp1 = (pp1[0:2] / pp1[2]) - frame1.kpus[idx1[i]]
    pp2 = (pp2[0:2] / pp2[2]) - frame2.kpus[idx2[i]]
    pp1 = np.sum(pp1**2)
    pp2 = np.sum(pp2**2)
    if pp1 > proj_err_treshold or pp2 > proj_err_treshold:
      rip += 1
      continue
    pt = p[:3]
    pts3d.append(pt)

  for pt1, pt2 in zip(frame1.kps[idx1], frame2.kps[idx2]):
    u1, v1 = denormalize(K, pt1)
    u2, v2 = denormalize(K, pt2)
    cv2.circle(img, (u1, v1), color=(0,255,0), radius=1)
    cv2.line(img, (u1, v1), (u2, v2), color=(255, 255,0))

  camera_position = get_camera_position(frame2.pose)
  
  frame1.pts = pts3d

  return img, pts3d, camera_position