from typing_extensions import Self
from numpy import ndarray
from typing import Tuple

class Adam():
    def __init__(self, alpha: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, tol: float = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.tol = tol

def solve_gradient_descent(video: ndarray, alpha_squared: float, step: float, max_iter: int, tol: float, norm_l2: bool) -> Tuple[ndarray, ndarray, int]:
    ...