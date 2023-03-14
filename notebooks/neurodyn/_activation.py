"""Activation functions for the rate neurons"""

import numpy as np
from scipy.special import expit

__all__ = ['sigmoid', 'identity']

def identity(x):
	return x

def sigmoid(x):
	return expit(x)