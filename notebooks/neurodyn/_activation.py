"""Activation functions for the rate neurons"""

import numpy as np
from typing import Callable
from scipy.special import expit

__all__ = ['sigmoid', 'identity', 'linear']

def identity(x: float) -> float:
	return x

def linear(x, a: float = 1.0, b: float = 0.0) -> float:
	return a*x + b

def sigmoid(x: float) -> float:
	return expit(x)