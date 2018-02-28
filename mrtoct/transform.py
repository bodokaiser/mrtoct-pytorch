import numpy as np


class Pad:
  """Pads numpy array to target shape.

  Zero values in target shape will be leave the corresponding dimension
  unpadded.
  """

  def __init__(self, shape):
    self.shape = np.array(shape)

  def __call__(self, x):
    offset = (self.shape - np.array(x.shape)) / 2
    padding = np.array([(np.floor(off), np.ceil(off)) for off in offset])
    padding[padding < 0] = 0

    return np.pad(x, padding.astype(np.int), 'constant')
