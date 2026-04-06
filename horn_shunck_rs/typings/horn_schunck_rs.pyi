from numpy import ndarray
from typing import Tuple

def solve_gradient_descent(video: ndarray, alpha_squared: float, step: float, max_iter: int, norm_l2: bool) -> Tuple[ndarray, ndarray]:
    ...