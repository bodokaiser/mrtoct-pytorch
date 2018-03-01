import argparse
import numpy as np
import h5py

from mrtoct.dataset import NIFTI
from mrtoct.transform import Pad

from scipy.ndimage import gaussian_filter


def main(args):
  volume_dataset = NIFTI(args.volumes_path)
  mask_dataset = NIFTI(args.masks_path) if args.masks_path else None

  pad = Pad([0, args.target_height, args.target_width])

  volumes = []

  for i in range(len(volume_dataset)):
    volume = volume_dataset[i]

    if mask_dataset is not None:
      mask = mask_dataset[i].astype(np.float32)
      volume = np.multiply(volume, gaussian_filter(mask, 2))

    volumes.append(pad(volume))

  slices = np.concatenate(volumes)

  with h5py.File(args.target_path, 'w') as f:
    dataset = f.create_dataset('slices', data=slices)
    dataset.attrs['vmin'] = np.min(slices)
    dataset.attrs['vmax'] = np.max(slices)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--volumes-path', required=True)
  parser.add_argument('--target-path', required=True)
  parser.add_argument('--masks-path')
  parser.add_argument('--target-height', default=384)
  parser.add_argument('--target-width', default=384)

  main(parser.parse_args())
