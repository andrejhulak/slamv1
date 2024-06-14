import os
import numpy as np
import cv2
from frame import *
from extract_matches import *
from triangulation import *
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from PyQt5 import QtWidgets
from point import *

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
        self.line = GLLinePlotItem(width=3, color=(0, 0, 1, 1))

        self.view.addItem(self.scatter2)
        self.view.addItem(self.line)

        self.camera_path = []
        self.points3D_model = []

    def update_scatter(self, pts3D):
        # Update camera position

        
        # Update 3D points
        self.points3D_model.extend(pts3D)
        self.scatter2.setData(pos=np.array(self.points3D_model))

        self.view.update()



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = ScatterPlot3D()
    window.show()

    cap = cv2.VideoCapture('car3.mp4')
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
            #print(frame2.pose)
            good_pts4d = np.array([frame1.pts[i] is None for i in idx1])
            pts4d = triangulate(frame1.pose, frame2.pose, frame1.kps[idx1], frame2.kps[idx2])
            #pts4d = np.array(pts4d)
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
                #pt = Rt[:3, :3] @ p[:3] + Rt[:3, 3:]
                pt = p[:3]
                pts3d_model.append(pt)
                    
            for pt1, pt2 in zip(frame1.kps[idx1], frame2.kps[idx2]):
                u1, v1 = denormalize(K, pt1)
                u2, v2 = denormalize(K, pt2)
                cv2.circle(frame, (u1, v1), color=(0,255,0), radius=1)
                cv2.line(frame, (u1, v1), (u2, v2), color=(255, 255,0))

            pts3d_model = np.array(pts3d_model)
            window.update_scatter(pts3d_model)
            
            cv2.imshow('Slam', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()