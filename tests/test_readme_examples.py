import math

import matplotlib
import numpy as np

# matplotlib.use("Agg")
plot = False
rtol = 1e-9
atol = 1e-12

import numerical_ode_solver as nos


def run_generated_code(code: str):
    namespace = {}
    exec(code, namespace)
    return namespace


def test_package_exports_public_classes():
    assert nos.IVP is not None
    assert nos.BVP is not None


def test_ivp_readme_example_matches_analytic_solution():
    """
    Initial value problem:
        y''(t) + 2 * y(t) = 0

    Interval:
        t ∈ [0, 5]

    Initial state:
        y(0) = 1
        y'(0) = 0
        
    Solution:
        y(t) = cos(sqrt(2) * t)
    """

    ode = nos.IVP(
        odes="y''+2*y=0",
        interval=(0, 5),
        initial_state=(1, 0)
        )
    code = ode.generate_scipy_string(plot=plot, atol=atol, rtol=rtol)
    namespace = run_generated_code(code)

    solution = namespace["solution"]
    assert solution.success

    expected_y = np.cos(np.sqrt(2) * solution.t)
    expected_y_prime = -np.sqrt(2) * np.sin(np.sqrt(2) * solution.t)
    np.testing.assert_allclose(solution.y[0], expected_y, atol=atol*100, rtol=rtol*100)
    np.testing.assert_allclose(solution.y[1], expected_y_prime, atol=atol*100, rtol=rtol*100)


def test_bvp_parameter_example_solves_for_k():
    ode = nos.BVP(
        odes="y'' + y = k",
        interval=(0, 1),
        bcs=("y(0)=0", "y(1)=1", "y'(0)=0"),
        initial_guess=(0, 0),
        params="k",
    )
    code = ode.generate_scipy_string(plot=plot)
    namespace = run_generated_code(code)

    solution = namespace["solution"]
    assert solution.success

    expected_k = 1 / (1 - np.cos(1))
    assert math.isclose(float(namespace["k"]), expected_k, rel_tol=1e-4, abs_tol=1e-4)

    np.testing.assert_allclose(solution.y[0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.y[0, -1], 1.0, atol=1e-5)
    np.testing.assert_allclose(solution.y[1, 0], 0.0, atol=1e-5)


def test_bvp_system_example_matches_boundary_conditions():
    ode = nos.BVP(
        odes=("y'' + z'' = -np.sin(x)", "z'' - y = np.cos(x)"),
        interval=(0, 1),
        bcs=("y(0) = 0", "y(1) = 1", "z(0) = 0", "z(1) = 0"),
    )
    code = ode.generate_scipy_string(plot=plot)
    namespace = run_generated_code(code)

    solution = namespace["solution"]
    assert solution.success

    np.testing.assert_allclose(solution.y[0, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.y[0, -1], 1.0, atol=1e-5)
    np.testing.assert_allclose(solution.y[2, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(solution.y[2, -1], 0.0, atol=1e-5)


if __name__ == "__main__":
    test_ivp_readme_example_matches_analytic_solution()
    test_bvp_parameter_example_solves_for_k()
    test_bvp_system_example_matches_boundary_conditions()