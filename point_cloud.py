import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLScatterPlotItem, GLLinePlotItem
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSize
import numpy as np

class ScatterPlot3D(QtWidgets.QMainWindow):
  def __init__(self):
    super().__init__()
    self.view = GLViewWidget()
    self.setCentralWidget(self.view)
    self.view.opts['distance'] = 200

    self.scatter2 = GLScatterPlotItem(size=1, color=(1, 1, 1, 0.4), pxMode=True)
    self.line = GLLinePlotItem(width=5, color=(0, 0, 1, 1))

    self.view.addItem(self.scatter2)
    self.view.addItem(self.line)

    self.camera_path = []
    self.points3D_model = []

  def update_scatter(self, pts3D, camera_position):
    self.camera_path.append(camera_position)
    self.line.setData(pos=np.array(self.camera_path))

    self.points3D_model.extend(pts3D)
    self.scatter2.setData(pos=np.array(self.points3D_model))

    self.view.setCameraPosition(pos=camera_position)
    self.view.opts['center'] = pg.Vector(camera_position)

    self.view.update()

def distance_filtering(points, camera_position, max_distance):
  distances = np.linalg.norm(points - camera_position, axis=1)
  return points[distances < max_distance]