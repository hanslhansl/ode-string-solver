# ode-string-solver

String-based ODE/BVP parsing and solving on top of SymPy + SciPy.

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
    equations=["y''(t) + c*y'(t) + k*y(t) = 0"],
    initial_conditions=["y(t0)=1", "y'(t0)=0"],
)

sol = problem.solve(t_span=(0.0, 5.0), namespace={"c": 0.3, "k": 2.0})
```

IC/BC points must use boundary symbols (defaults: `t0` for IVP, `a`/`b` for BVP). Provide any constants or functions referenced in the ODE via the `namespace` argument when solving.

```python
from ode_string_solver import BVPProblem
import numpy as np

problem = BVPProblem.from_strings(
    equations=["y''(x) + y(x) = 0"],
    boundary_conditions=["y(a)=0", "y(b)=1"],
    initial_guess=["x", "1"],
)

sol = problem.solve(x_mesh=np.linspace(0.0, np.pi / 2.0, 41))
```
