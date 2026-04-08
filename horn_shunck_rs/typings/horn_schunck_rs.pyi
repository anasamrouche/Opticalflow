from typing_extensions import Self
from numpy import ndarray
from typing import Tuple

def solve_gradient_descent(video: ndarray, alpha_squared: float, step: float, max_iter: int, tol: float, norm_l2: bool) -> Tuple[ndarray, ndarray, int]:
    ...

def solve_gauss_seidel(video: ndarray, alpha_squared, max_iter) -> Tuple[ndarray, ndarray]:
    ...