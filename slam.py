import os
import numpy as np
import cv2
from frame import *
from extract_matches import *
from triangulation import *
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from PyQt5 import QtWidgets

#F= int(os.getenv("F","500"))
W, H = 1920//2, 1080//2
#W, H = 1920, 1080
F = 270
imgsize = (W, H)
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
frames = []

class ScatterPlot3D(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.view = GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.opts['distance'] = 40

        self.scatter2 = GLScatterPlotItem(size=1, color=(1, 1, 1, 0.6), pxMode=True)
        self.line = GLLinePlotItem(width=5, color=(0, 0, 1, 1))

        self.view.addItem(self.scatter2)
        self.view.addItem(self.line)

        self.camera_path = []
        self.points3D_model = []

    def update_scatter(self, pts3D, camera_position):
        # Update camera position
        self.camera_path.append(camera_position)
        self.line.setData(pos=np.array(self.camera_path))

        # Update 3D points
        self.points3D_model.extend(pts3D)
        self.scatter2.setData(pos=np.array(self.points3D_model))

        self.view.update()

def get_camera_position(pose):
    R = pose[0:3, 0:3]
    t = pose[0:3, 3]
    camera_position = -np.dot(R.T, t)
    return camera_position

def random_sampling(points, num_samples):
    if len(points) <= num_samples:
        return points
    return points[np.random.choice(points.shape[0], num_samples, replace=False)]

def distance_filtering(points, camera_position, max_distance):
    distances = np.linalg.norm(points - camera_position, axis=1)
    return points[distances < max_distance]

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ScatterPlot3D()
    window.show()

    cap = cv2.VideoCapture('car1.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, imgsize)
            slam_frame = Frame(frame, K, imgsize)
            frames.append(slam_frame)
            if slam_frame.id == 0:
                continue
            frame1 = frames[-2]
            frame2 = frames[-1]
            idx1, idx2, Rt = generate_match(frame1, frame2)
            if frame2.id <= 2:
                continue

            frame2.pose = np.dot(Rt, frame1.pose)
            good_pts4d = np.array([frame1.pts[i] is None for i in idx1])
            pts4d = cv2.triangulatePoints(frame1.pose[:3, :], frame2.pose[:3, :], frame1.kps[idx1].T, frame2.kps[idx2].T).T
            pts4d /= pts4d[:, 3:]
            #print("Adding:  %d points" % np.sum(unmatched_points))

            pts3d_model = []

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
                if pp1 > 2 or pp2 > 2:
                    continue
                pt = p[:3]
                pts3d_model.append(pt)
                    
            for pt1, pt2 in zip(frame1.kps[idx1], frame2.kps[idx2]):
                u1, v1 = denormalize(K, pt1)
                u2, v2 = denormalize(K, pt2)
                cv2.circle(frame, (u1, v1), color=(0,255,0), radius=1)
                cv2.line(frame, (u1, v1), (u2, v2), color=(255, 255,0))

            camera_position = get_camera_position(frame2.pose)
            # apply random sampling
            num_samples = 100
            pts3d_model = random_sampling(np.array(pts3d_model), num_samples)

            # apply distance filtering
            max_distance = 50
            pts3d_model = distance_filtering(np.array(pts3d_model), camera_position, max_distance)

            window.update_scatter(pts3d_model, camera_position)
            
            cv2.imshow('Slam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()