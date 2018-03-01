import unittest
import torch
import numpy as np

from mrtoct import transform


class TestPad(unittest.TestCase):

  def setUp(self):
    self.t1 = transform.Pad([160])
    self.t2 = transform.Pad([160, 200])
    self.t3 = transform.Pad([160, 200, 120])
    self.t4 = transform.Pad([160, 0, 120])

  def test_transform(self):
    x1 = np.random.randn(120)
    x2 = np.random.randn(120, 140)
    x3 = np.random.randn(120, 140, 100)

    y1 = np.pad(x1, [(20, 20)], 'constant')
    y2 = np.pad(x2, [(20, 20), (30, 30)], 'constant')
    y3 = np.pad(x3, [(20, 20), (30, 30), (10, 10)], 'constant')
    y4 = np.pad(x3, [(20, 20), (0, 0), (10, 10)], 'constant')

    np.testing.assert_array_equal(self.t1(x1), y1)
    np.testing.assert_array_equal(self.t2(x2), y2)
    np.testing.assert_array_equal(self.t3(x3), y3)
    np.testing.assert_array_equal(self.t4(x3), y4)


class TestNormalize(unittest.TestCase):

  def setUp(self):
    self.t1 = transform.Normalize(100)
    self.t2 = transform.Normalize(100, 10)

  def test_transform(self):
    x1 = np.random.randn(120)
    x2 = np.random.randn(120)

    y1 = x1 / 100
    y2 = (x2 - 10) / 90

    np.testing.assert_array_equal(self.t1(x1), y1)
    np.testing.assert_array_equal(self.t2(x2), y2)


class TestToTensor(unittest.TestCase):

  def setUp(self):
    self.t = transform.ToTensor()

  def test_transform(self):
    x = np.random.randn(120, 200)
    y = np.expand_dims(x, 0).astype(np.float32)

    self.assertIsInstance(self.t(x), torch.Tensor)
    np.testing.assert_array_equal(self.t(x).numpy(), y)
