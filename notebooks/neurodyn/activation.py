"""Activation functions for the rate neurons"""

import numpy as np
from scipy.special import expit as sigmoid

__all__ = ['sigmoid', 'identity']

def identity(x):
	return x