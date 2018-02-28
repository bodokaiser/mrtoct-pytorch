import argparse
import numpy as np
import h5py

from mrtoct.dataset import NIFTI
from mrtoct.transform import Pad


def convert(inputs_path, output_path, height=384, width=384):
  transform = Pad([0, height, width])

  slices = np.concatenate(NIFTI(inputs_path, transform=transform))

  with h5py.File(output_path, 'w') as f:
    dataset = f.create_dataset('slices', data=slices)
    dataset.attrs['vmin'] = np.min(slices)
    dataset.attrs['vmax'] = np.max(slices)


def main(args):
  convert(args.inputs_path, args.output_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path')
  parser.add_argument('--output-path')

  main(parser.parse_args())
