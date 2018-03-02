import os
import h5py
import nibabel as nb
import numpy as np

from torch.utils.data import Dataset

from mrtoct.transform import Normalize


def is_nifti(filename):
  return any(filename.endswith(ext) for ext in ['.nii', '.nii.gz'])


class NIFTI(Dataset):

  def __init__(self, root, transform=None):
    self.transform = transform
    self.filenames = [os.path.join(root, f)
                      for f in os.listdir(root) if is_nifti(f)]
    self.filenames.sort()

  def __getitem__(self, index):
    if index >= len(self.filenames):
      raise IndexError

    volume = nb.load(self.filenames[index]).get_data()
    volume = np.transpose(volume)
    volume = np.flip(volume, 0)
    volume = np.flip(volume, 1)

    if self.transform is not None:
      volume = self.transform(volume)

    return volume

  def __len__(self):
    return len(self.filenames)


class HDF5(Dataset):

  def __init__(self, path, field):
    f = h5py.File(path, 'r')

    self.dataset = f[field]
    assert len(self.inputs) == len(self.targets)

  def __getitem__(self, index):
    if index >= len(self):
      raise IndexError

    return self.dataset[index]

  def __getattr__(self, name):
    return self.dataset.attrs[name]

  def __len__(self):
    return len(self.dataset)


class Patch(Dataset):

  def __init__(self, path, transform=None, target_transform=None):
    self.inputs = HDF5(path, 'inputs')
    self.targets = HDF5(path, 'targets')
    assert len(self.inputs) == len(self.targets)

    self.input_norm = Normalize(self.inputs.vmax, self.inputs.vmin)
    self.target_norm = Normalize(self.targets.vmax, self.targets.vmax)

  def __getitem__(self, index):
    input = self.input_norm(self.targets[index])
    target = self.target_norm(self.inputs[index])

    if self.transform is not None:
      input = self.transform(input)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return input, target
