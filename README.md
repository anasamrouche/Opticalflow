# Opticalflow

**Fast Horn–Schunck optical flow in Rust, with Python bindings.**

Opticalflow is a Rust module, exposed to Python through [PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs), that estimates motion in video streams by minimizing the Horn–Schunck energy functional. Several solvers are implemented and can be compared side by side: gradient descent (L1 and L2), Gauss–Seidel, and coarse-to-fine pyramidal Gauss–Seidel. A GPU-accelerated real-time mode is also available.

https://github.com/user-attachments/assets/c8859cba-5cef-43e3-8701-8a0c247ca51f

---

## Table of contents

- [Background](#background)
- [Supported methods](#supported-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Real-time detection](#real-time-detection)
- [References](#references)

---

## Background

Given a sequence of grayscale frames $I(x, y, t)$, optical flow estimates a dense velocity field $(u, v)$ describing the apparent motion of each pixel.

Horn and Schunck (1981) combine two assumptions:

1. **Brightness constancy** — a moving point keeps its intensity, which linearizes to
   $I_x u + I_y v + I_t = 0$.
2. **Smoothness** — neighboring pixels move similarly, so the flow field should have small spatial gradients.

This leads to the minimization of the energy functional

$$
E(u, v) = \iint_\Omega \Big( I_x u + I_y v + I_t \Big)^2 + \alpha^2 \Big( \lVert \nabla u \rVert^2 + \lVert \nabla v \rVert^2 \Big) \, dx \, dy
$$

where $\alpha$ controls the trade-off between fidelity to the data and smoothness of the flow.

The associated Euler–Lagrange equations yield the classical fixed-point update, where $\bar u, \bar v$ denote local averages of the flow:

$$
u \leftarrow \bar u - \frac{I_x \left( I_x \bar u + I_y \bar v + I_t \right)}{\alpha^2 + I_x^2 + I_y^2},
\qquad
v \leftarrow \bar v - \frac{I_y \left( I_x \bar u + I_y \bar v + I_t \right)}{\alpha^2 + I_x^2 + I_y^2}
$$

Opticalflow also provides an **L1 variant** of the functional, which replaces the quadratic penalty with an absolute-value penalty. It is more robust to outliers (occlusions, noise, lighting changes) and better preserves motion discontinuities, at the cost of a non-smooth optimization problem.

---

## Supported methods

| Solver                    | L2 functional | L1 functional | Notes                                                        |
|---------------------------|:-------------:|:-------------:|--------------------------------------------------------------|
| Gradient descent          | ✅            | ✅            | Direct descent on the energy; only solver supporting L1      |
| Gauss–Seidel              | ✅            | —             | In-place fixed-point iterations on the Euler–Lagrange system |
| Pyramidal Gauss–Seidel    | ✅            | —             | Coarse-to-fine scheme, handles large displacements           |

The Gauss–Seidel solvers currently only optimize the L2 functional.

---

## Installation

### Prerequisites

- Python ≥ 3.11

 with a virtual environment
- The Rust toolchain

**Unix (Linux / macOS)**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**Windows**

Download and run the installer from [rustup.rs](https://rustup.rs/).

### Build

From the project root, inside your virtual environment:

```bash
pip install maturin
maturin develop --release
```

`maturin develop` compiles the crate (Cargo resolves the Rust dependencies from `Cargo.toml`) and installs the resulting Python module directly into the active environment. Always build with `--release`: debug builds are orders of magnitude slower for numerical code.

To produce a distributable wheel instead:

```bash
maturin build --release
```

---

## Usage

Install the Python dependencies of the final script in your virtual environment, then run it. It processes the test videos with every available solver and writes the results as follows:

```
.
└── tests/
    ├── norm_L1/
    │   └── gradient_results/
    └── norm_L2/
        ├── gradient_results/
        ├── gauss_seidel_results/
        └── pyramidal_gauss_seidel_results/
```

---

## Real-time detection

Opticalflow includes a real-time motion detection mode that leverages GPU acceleration:

```python
import opticalflow

opticalflow.real_time_detection()
```

---

## References

- B. K. P. Horn and B. G. Schunck, *Determining Optical Flow*, Artificial Intelligence, 17(1–3), 185–203, 1981.
- A. Bruhn, J. Weickert, C. Schnörr, *Lucas/Kanade Meets Horn/Schunck: Combining Local and Global Optic Flow Methods*, International Journal of Computer Vision, 61(3), 211–231, 2005.
