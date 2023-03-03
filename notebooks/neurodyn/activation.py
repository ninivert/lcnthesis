import numpy as np
from scipy.special import expit as sigmoid

__all__ = ['sigmoid', 'identity']

def identity(x):
	return x