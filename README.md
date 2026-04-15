# ode-string-solver

String-based ODE/BVP parsing on top of SymPy + SciPy.

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

Run them with:

```bash
uv run python examples/ivp_example.py
uv run python examples/bvp_example.py
```

Tests in `tests/` are for verification and regression protection.
They are not a substitute for usage examples.

## Basic API

```python
from ode_string_solver import prepare_ivp_problem, solve_ivp_from_problem

problem = prepare_ivp_problem(
    equations=["d2 y / dt2 + c*dy/dt + k*y(t) = 0"],
    initial_conditions=["y(0)=1", "y'(0)=0"],
)
sol = solve_ivp_from_problem(problem, t_span=(0.0, 5.0), params={"c": 0.3, "k": 2.0})
```
