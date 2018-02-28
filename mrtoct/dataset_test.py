import unittest

from mrtoct import dataset


class TestNIFTI(unittest.TestCase):

  def setUp(self):
    self.training = dataset.NIFTI('data')
    self.validation = dataset.NIFTI('data', train=False)

  def testLength(self):
    self.assertEqual(13, len(self.training))
    self.assertEqual(4, len(self.validation))

  def testGetItem(self):
    mr, ct = self.training[0]

    self.assertTupleEqual(mr.shape, ct.shape)
