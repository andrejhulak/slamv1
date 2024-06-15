import os
import numpy as np
import cv2
from frame import *
from extract_matches import *
from triangulation import *
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from PyQt5 import QtWidgets
from time import sleep

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

        self.scatter2 = GLScatterPlotItem(size=1, color=(1, 1, 1, 0.4), pxMode=True)
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

    cap = cv2.VideoCapture('videos/car3.mp4')
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

            img, pts3d, camera_position = triangulate(frame, frame1, frame2, Rt, idx1, idx2)

            if len(pts3d) == 0:
                continue

            max_dist = 100
            pts3d = distance_filtering(np.array(pts3d), camera_position, max_dist)
            
            window.update_scatter(pts3d, camera_position)
            
            #sleep(0.5)

            cv2.imshow('Slam', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()