import argparse
import numpy as np
import h5py

from mrtoct.dataset import NIFTI
from mrtoct.transform import Pad


def convert(inputs_path, targets_path, output, height=384, width=384):
  transform = Pad([0, height, width])

  inputs = np.concatenate(NIFTI(inputs_path, transform=transform))
  targets = np.concatenate(NIFTI(targets_path, transform=transform))

  with h5py.File(output, 'w') as f:
    dataset = f.create_dataset('inputs', data=inputs)
    dataset.attrs['vmin'] = np.min(inputs)
    dataset.attrs['vmax'] = np.max(inputs)

    dataset = f.create_dataset('targets', data=targets)
    dataset.attrs['vmin'] = np.min(targets)
    dataset.attrs['vmax'] = np.max(targets)


def main(args):
  convert(args.inputs_path, args.targets_path, args.output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path')
  parser.add_argument('--targets-path')
  parser.add_argument('output')

  main(parser.parse_args())
