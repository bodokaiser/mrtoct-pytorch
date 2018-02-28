import unittest

from mrtoct import dataset


class TestNIFTI(unittest.TestCase):

  def setUp(self):
    self.mr_dataset = dataset.NIFTI('data/training/mr')
    self.ct_dataset = dataset.NIFTI('data/training/ct')

  def test_length(self):
    self.assertEqual(13, len(self.mr_dataset))
    self.assertEqual(13, len(self.ct_dataset))

  def test_get_item(self):
    mr = self.mr_dataset[0]
    ct = self.ct_dataset[0]

    self.assertTupleEqual(mr.shape, ct.shape)
