# ode-string-solver

String-based ODE/BVP parsing on top of SymPy + SciPy.

## Install from Git

```bash
python -m pip install "git+https://github.com/hanslhansl/ode-string-solver.git"
```

## Setup with uv

```bash
uv sync
```

## Run tests

```bash
uv run pytest
```

## Where examples live

User-facing examples are in `examples/`:

- `examples/ivp_example.py`
- `examples/bvp_example.py`
- `examples/generate_script_example.py` (only generates Python solver strings; does not solve)

Run them with:

```bash
uv run python examples/ivp_example.py
uv run python examples/bvp_example.py
uv run python examples/generate_script_example.py
```

Tests for verification and regression protection are located in `tests/`.

## API

```python
from ode_string_solver import IVPProblem

problem = IVPProblem.from_strings(
    equations=["d2 y / dt2 + c*dy/dt + k*y(t) = 0"],
    initial_conditions=["y(0)=1", "y'(0)=0"],
)

sol = problem.solve(t_span=(0.0, 5.0), params={"c": 0.3, "k": 2.0})
```

```python
from ode_string_solver import BVPProblem
import numpy as np

problem = BVPProblem.from_strings(
    equations=["d2y/dx2 + y(x) = 0"],
    boundary_conditions=["y(0)=0", "y(1.5707963267948966)=1"],
    initial_guess=["x", "1"],
    left_boundary="0",
    right_boundary="1.5707963267948966",
)

sol = problem.solve(x_mesh=np.linspace(0.0, np.pi / 2.0, 41))
```
