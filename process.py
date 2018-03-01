import argparse
import numpy as np
import h5py

from mrtoct.dataset import NIFTI
from mrtoct.transform import Pad

from scipy.ndimage import gaussian_filter


def main(args):
  transform = Pad([0, args.target_height, args.target_width])

  volumes = []

  for volume, mask in zip(NIFTI(args.inputs_path), NIFTI(args.masks_path)):
    mask = gaussian_filter(mask.astype(np.float32), 2)

    volumes.append(transform(np.multiply(volume, mask)))

  slices = np.concatenate(volumes)

  with h5py.File(args.output_path, 'w') as f:
    dataset = f.create_dataset('slices', data=slices)
    dataset.attrs['vmin'] = np.min(slices)
    dataset.attrs['vmax'] = np.max(slices)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--masks-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--target-height', default=384)
  parser.add_argument('--target-width', default=384)

  main(parser.parse_args())
