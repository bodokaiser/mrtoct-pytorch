import os
import h5py
import nibabel as nb
import numpy as np

from torch.utils.data import Dataset


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

  def __init__(self, path, name, transform=None):
    f = h5py.File(path, 'r')
    self.dset = f[name]

    self.meta = {}
    for k, v in self.dset.attrs.items():
      self.meta[k] = v

    self.transform = transform

  def __getitem__(self, index):
    if index >= len(self.dset):
      raise IndexError

    return self.dset[index]

  def __len__(self):
    return len(self.dset)


class Combined(Dataset):

  def __init__(self, dataset1, dataset2, transform=None,
               target_transform=None):
    self.dataset1 = dataset1
    self.dataset2 = dataset2
    assert len(self.dataset1) == len(self.dataset2)

    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    x = self.dataset1[index]
    y = self.dataset2[index]

    if self.transform is not None:
      x = self.transform(x)
    if self.target_transform is not None:
      y = self.target_transform(y)

    return x, y

  def __len__(self):
    return len(self.dataset1)
