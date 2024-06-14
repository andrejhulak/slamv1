class Point(object):
  def __init__(self, loc):
    self.pt = loc
    self.frames = []
    self.idxs = []

  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)