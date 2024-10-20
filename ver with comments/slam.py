import os
import numpy as np
import cv2
from frame import *
from extract_matches import *
from triangulation import *
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from PyQt5 import QtWidgets

F= int(os.getenv("F","500"))
W, H = 1920//2, 1080//2
imgsize = (W, H)
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
frames = []
camera_model_pos = np.ones((3, 1))

class ScatterPlot3D(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.view = GLViewWidget()
        self.setCentralWidget(self.view)
        self.view.opts['distance'] = 40

        self.scatter1 = GLScatterPlotItem(size=5, color=(1, 0, 0, 1), pxMode=True)
        self.scatter2 = GLScatterPlotItem(size=1, color=(0, 1, 0, 1), pxMode=True)
        self.line = GLLinePlotItem(width=1, color=(0, 0, 1, 1))

        self.view.addItem(self.scatter1)
        self.view.addItem(self.scatter2)
        self.view.addItem(self.line)

        self.camera_path = []
        self.points3D_model = []

    def update_scatter(self, cam_pos, pts3D):
        self.scatter1.setData(pos=cam_pos)
            
        #self.points3D_model.extend(pts3D)
        self.points3D_model = pts3D
        self.scatter2.setData(pos=np.array(self.points3D_model))
        
        self.camera_path.append(cam_pos.flatten())
        self.line.setData(pos=np.array(self.camera_path))
        
        self.view.update()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ScatterPlot3D()
    window.show()
    #T_n = np.eye(4)

    cap = cv2.VideoCapture('car1.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(frame, imgsize)
            slam_frame = Frame(frame, K, imgsize)
            frames.append(slam_frame)
            if slam_frame.id == 0:
                continue
            pts_disp = []
            frame1 = frames[-2]
            frame2 = frames[-1]
            idx1, idx2, Rt = generate_match(frame1, frame2)
            # R = Rt[:3, :3]
            # t = Rt[:3, 3:]
            # mat = np.zeros((4, 3))
            # mat[:3, :3] = R 
            # mat2 = np.ones((4, 1))
            # mat2[:3, :3] = t
            # T_n = T_n @ np.append(mat, mat2, axis=1)
            # R_n = T_n[:3, :3]
            # t_n = T_n[:3, 3:]
            # frame2.pose = K @ np.append(R_n.T, -np.dot(R_n.T, t_n), axis=1)

            if frame2.id <= 3:
                continue

            frame1.pose =np.dot(Rt,frame2.pose)
            
            pts4d = triangulate(frame1.pose, frame2.pose, frame1.kps[idx1], frame2.kps[idx2])
            #pts4d = cv2.triangulatePoints(frame1.pose[:3, :], frame2.pose[:3, :], frame1.kps[idx1].T, frame2.kps[idx2].T).T
            #pts4d = cv2.triangulatePoints(frame1.pose, frame2.pose, frame1.kps[idx1].T, frame2.kps[idx2].T).T
            pts4d /= pts4d[:, 3:]
            pts3d = pts4d[:, :3]
            unmatched_points = np.array([frame1.pts[i] is None for i in idx1])
            #print("Adding:  %d points" % np.sum(unmatched_points))
            good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points
            for i,p in enumerate(pts4d):
                if not good_pts4d[i]:
                    continue
                pts_disp.append(denormalize(frame2.Kinv, p))
                pts_disp.append(p[:3])
            #camera_model_pos = Rt[:3, :3] @ camera_model_pos + Rt[:3, 3:]
            #window.update_scatter(camera_model_pos.T, pts_disp)

            for pt1, pt2 in zip(frame1.kps[idx1], frame2.kps[idx2]):
                u1, v1 = denormalize(K, pt1)
                u2, v2 = denormalize(K, pt2)
                cv2.circle(frame, (u1, v1), color=(0,255,0), radius=1)
                cv2.line(frame, (u1, v1), (u2, v2), color=(255, 255,0))
            
            
            cv2.imshow('Slam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()