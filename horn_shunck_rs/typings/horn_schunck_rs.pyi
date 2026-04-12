from typing_extensions import Self
from numpy import ndarray
from typing import Tuple, List

def solve_gradient_descent(video: ndarray, alpha_squared: float, step: float, max_iter: int, tol: float, norm_l2: bool) -> Tuple[ndarray, ndarray, ndarray]:
    ...

def solve_gauss_seidel(video: ndarray, alpha_squared: float, max_iter: int) -> Tuple[ndarray, ndarray]:
    ...

def pyramidal_gauss_seidel(video: ndarray, alpha_squared: float, max_iter: int, recursion_depth: int) -> Tuple[ndarray, ndarray]:
    ...