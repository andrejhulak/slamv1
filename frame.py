import numpy as np
import cv2

CULLING_ERR_THRES = 0.02

frame_id = 0

def normalize(Kinv, pts):
    return np.dot(Kinv, np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T[:, 0:2]

def denormalize(count, pt):
    ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def feature_matching(frame):
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 1000, qualityLevel=0.01, minDistance=7)
    kps = np.array([cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts])
    kps, des = orb.compute(frame, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

class Frame(object):
    def __init__(self, img, cam_intr_mat, imgsize):
        global frame_id
        self.id = frame_id
        self.K = cam_intr_mat
        self.Kinv = np.linalg.inv(self.K)
        self.w, self.h = imgsize
        self.kpus, self.des = feature_matching(img)
        self.kps = normalize(self.Kinv, self.kpus)
        self.pts = [None]*len(self.kps)
        self.pose = np.eye(4)
        frame_id += 1
