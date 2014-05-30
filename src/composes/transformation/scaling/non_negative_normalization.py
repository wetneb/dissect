'''
Created on May 30, 2014

@author: Antonin Delpeuch
'''

from scaling import Scaling

class NonNegativeNormalization(Scaling):
    """
    Sets any negative coefficient in a space to zero.

    """
    _name = "non_negative_normalization"

    def apply(self, matrix_):
        matrix_.to_non_negative()
        return matrix_


