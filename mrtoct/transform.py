import torch
import numpy as np


class Pad:
  """Pads numpy array to target shape.

  Zero values in target shape will be leave the corresponding dimension
  unpadded.

  Args:
    shape: tuple or list with target shape
  """

  def __init__(self, shape):
    self.shape = np.array(shape)

  def __call__(self, x):
    offset = (self.shape - np.array(x.shape)) / 2
    padding = np.array([(np.floor(off), np.ceil(off)) for off in offset])
    padding = np.where(padding < 0, 0, padding).astype(np.int)

    return np.pad(x, padding, 'constant')


class Normalize:
  """Normalizes numpy array by constant value to range [0,1].

  Args:
    vmax: maximum value in dataset
    vmin: minimum value in dataset (default: 0)
  """

  def __init__(self, vmax, vmin=0):
    self.vmax = vmax
    self.vmin = vmin

  def __call__(self, x):
    return (x - self.vmin) / (self.vmax - self.vmin)


class ToTensor:
  """Converts numpy array to pytorch tensor."""

  def __call__(self, x):
    x = np.expand_dims(x, 0)

    return torch.from_numpy(x).float()
