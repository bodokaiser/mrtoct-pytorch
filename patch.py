import argparse
import h5py
import numpy as np

from mrtoct.dataset import NIFTI

from sklearn.feature_extraction.image import extract_patches_2d


def main(args):
  shape = (args.height, args.width)
  inputs = NIFTI(args.inputs_path)
  targets = NIFTI(args.targets_path)

  with h5py.File(args.filename, 'w') as f:
    dshape = (0,) + shape
    mshape = (None, ) + shape

    dsets = [
        f.create_dataset('inputs', dshape, maxshape=mshape,
                         compression='gzip'),
        f.create_dataset('targets', dshape, maxshape=mshape,
                         compression='gzip')
    ]

    for input, target in zip(inputs, targets):
      volume = np.stack([input, target], -1)

      for slice in volume:
        patches = extract_patches_2d(slice, shape, args.number)
        patches = patches[np.sum(patches[:, :, :, 0], axis=(1, 2)) > 0]
        patches = patches[np.sum(patches[:, :, :, 1], axis=(1, 2)) > 0]

        offset = dsets[0].len()
        length = len(patches)

        for i, dset in enumerate(dsets):
          dset.resize((offset + length,) + shape)
          dset[offset:] = patches[:, :, :, i]

    for dset in dsets:
      dset.attrs['vmin'] = np.min(dset)
      dset.attrs['vmax'] = np.max(dset)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--inputs-path', required=True)
  parser.add_argument('--targets-path', required=True)
  parser.add_argument('--height', required=True, type=int)
  parser.add_argument('--width', required=True, type=int)
  parser.add_argument('--number', default=100, type=int)
  parser.add_argument('filename')

  main(parser.parse_args())
