import unittest
import numpy as np
import keras
import keras.backend as K
from keras.utils import to_categorical

from image_models.losses.losses import *

def random_data(shape, num_classes):
    """
    Create some random data of a certain shape 
    """
    assert(len(shape) > 1)
    assert(num_classes >= 1)

    y = np.random.randint(num_classes, size=shape[:-1])
    return to_categorical(y, num_classes)

class TestDiceCoeff(unittest.TestCase):
    def setUp(self):
        self.shape_2d = (5, 100, 100, 1)
        self.shape_3d = (5, 10, 100, 100, 1)

    def test_2d_random_data_ranges(self):
        y_pred = random_data(self.shape_2d, 1)
        y_true = random_data(self.shape_2d, 1)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 1)

    def test_2d_matching_data(self):
        y_pred = random_data(self.shape_2d, 1)
        y_true = y_pred

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)


        self.assertEqual(s, 1)
    
    def test_2d_wrong_data(self):
        y_pred = np.ones(self.shape_2d)
        y_true = np.zeros(self.shape_2d)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 0)
    
    def test_3d_random_data_ranges(self):
        y_pred = random_data(self.shape_3d, 1)
        y_true = random_data(self.shape_3d, 1)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 1)

    def test_3d_matching_data(self):
        y_pred = random_data(self.shape_3d, 1)
        y_true = y_pred

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 1)
    
    def test_3d_wrong_data(self):
        y_pred = np.ones(self.shape_3d)
        y_true = np.zeros(self.shape_3d)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 0)


class TestGeneralizedDiceCoeff(unittest.TestCase):
    def setUp(self):
        self.shape_2d = (5, 100, 100, 4)
        self.shape_3d = (5, 10, 100, 100, 4)
    
    def test_2d_random_data_ranges(self):
        y_pred = random_data(self.shape_2d, 1)
        y_true = random_data(self.shape_2d, 1)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 1)

    def test_2d_matching_data(self):
        y_pred = random_data(self.shape_2d, 1)
        y_true = y_pred

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 1)
    
    def test_2d_wrong_data(self):
        y_pred = np.ones(self.shape_2d)
        y_true = np.zeros(self.shape_2d)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 0)
    
    def test_3d_random_data_ranges(self):
        y_pred = random_data(self.shape_3d, 1)
        y_true = random_data(self.shape_3d, 1)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertGreaterEqual(s, 0)
        self.assertLessEqual(s, 1)

    def test_3d_matching_data(self):
        y_pred = random_data(self.shape_3d, 1)
        y_true = y_pred

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 1)
    
    def test_3d_wrong_data(self):
        y_pred = np.ones(self.shape_3d)
        y_true = np.zeros(self.shape_3d)

        s = dice_coeff(y_pred, y_true)
        s = K.get_value(s)

        self.assertEqual(s, 0)
