import os
import nibabel as nb

from torch.utils.data import Dataset


EXTENSIONS = ['.nii', '.nii.gz']


def is_nifti(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS)


class NIFTI(Dataset):

  def __init__(self, root, train=True, transform=None, target_transform=None):
    prefix = 'training' if train else 'validation'
    self.mr_root = os.path.join(root, prefix, 'mr')
    self.ct_root = os.path.join(root, prefix, 'ct')

    self.mr_filenames = [f for f in os.listdir(self.mr_root) if is_nifti(f)]
    self.ct_filenames = [f for f in os.listdir(self.ct_root) if is_nifti(f)]
    assert len(self.mr_filenames) == len(self.ct_filenames)

    self.mr_filenames.sort()
    self.ct_filenames.sort()

    self.transform = transform
    self.target_transform = target_transform

  def __getitem__(self, index):
    mr_path = os.path.join(self.mr_root, self.mr_filenames[index])
    ct_path = os.path.join(self.ct_root, self.ct_filenames[index])

    mr = nb.load(mr_path).get_data()
    ct = nb.load(ct_path).get_data()

    if self.transform is not None:
      mr = self.transform(mr)
    if self.target_transform is not None:
      ct = self.target_transform(ct)

    return mr, ct

  def __len__(self):
    return len(self.mr_filenames)
