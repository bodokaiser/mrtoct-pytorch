import unittest
import numpy as np

from mrtoct import dataset


class TestNIFTI(unittest.TestCase):

  def setUp(self):
    self.mr_dataset = dataset.NIFTI('data/training/mr')
    self.ct_dataset = dataset.NIFTI('data/training/ct')

  def test_length(self):
    self.assertEqual(13, len(self.mr_dataset))
    self.assertEqual(13, len(self.ct_dataset))

  def test_getitem(self):
    mr = self.mr_dataset[0]
    ct = self.ct_dataset[0]

    self.assertTupleEqual(mr.shape, ct.shape)


class TestHDF5(unittest.TestCase):

  def setUp(self):
    self.dataset = dataset.HDF5('data/training/mr.h5', 'slices')

  def test_length(self):
    self.assertGreater(len(self.dataset), 0)

  def test_getattr(self):
    self.assertEqual(self.dataset.meta['vmin'], 0)

  def test_getitem(self):
    self.assertGreaterEqual(np.min(self.dataset[0]), 0)
