import numpy as np
import pytest

from ode_string_solver import IVPProblem


def test_ivp_prepare_and_callable_eval():
    problem = IVPProblem.from_strings(
        equations=[
            "y''(t) + c*y'(t) + k*y(t) = 0",
            "z'(t) - y(t) = 0",
        ],
        initial_conditions=[
            "y(t0) = 1",
            "y'(t0) = 0",
            "z(t0) = 2",
        ],
    )

    model = problem.build_callable(namespace={"c": 0.3, "k": 4.0})
    out = model.fun(0.0, [1.0, 0.0, 2.0])

    assert out.shape == (3,)
    assert out[0] == 0.0
    assert out[1] == -4.0
    assert out[2] == 1.0


def test_ivp_solve_smoke():
    problem = IVPProblem.from_strings(
        equations=["y''(t) + y(t) = 0"],
        initial_conditions=["y(t0) = 1", "y'(t0) = 0"],
    )
    sol = problem.solve(
        t_span=(0.0, 2.0),
        t_eval=np.linspace(0.0, 2.0, 81),
        rtol=1e-9,
        atol=1e-11,
    )

    assert sol.success
    assert sol.y.shape[0] == 2
    np.testing.assert_allclose(sol.y[0], np.cos(sol.t), rtol=2e-6, atol=2e-8)
    np.testing.assert_allclose(sol.y[1], -np.sin(sol.t), rtol=2e-6, atol=2e-8)


def test_ivp_callable_function_accessible_via_namespace():
    problem = IVPProblem.from_strings(
        equations=["y'(t) - a(t) = 0"],
        initial_conditions=["y(t0) = 0"],
    )

    def a(t: float) -> float:
        return 2.0

    sol = problem.solve(
        t_span=(0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 11),
        namespace={"a": a},
        rtol=1e-10,
        atol=1e-12,
    )

    assert sol.success
    np.testing.assert_allclose(sol.y[0], 2.0 * sol.t, rtol=1e-7, atol=1e-9)


def test_ivp_callable_and_builtin_math_function_work_together():
    problem = IVPProblem.from_strings(
        equations=["y'(t) - a(t)*cos(t) = 0"],
        initial_conditions=["y(t0) = 0"],
    )

    sol = problem.solve(
        t_span=(0.0, 1.0),
        t_eval=np.linspace(0.0, 1.0, 51),
        namespace={"a": lambda t: 1.0, "cos": np.cos},
        rtol=1e-10,
        atol=1e-12,
    )

    assert sol.success
    np.testing.assert_allclose(sol.y[0], np.sin(sol.t), rtol=2e-6, atol=2e-8)


def test_ivp_method_api_smoke():
    problem = IVPProblem.from_strings(
        equations=["y''(t) + y(t) = 0"],
        initial_conditions=["y(t0) = 1", "y'(t0) = 0"],
    )

    model = problem.build_callable()
    out = model.fun(0.0, [1.0, 0.0])
    assert out.shape == (2,)

    sol = problem.solve(
        t_span=(0.0, 2.0),
        t_eval=np.linspace(0.0, 2.0, 41),
        rtol=1e-9,
        atol=1e-11,
    )
    assert sol.success

    script = problem.generate_scipy_script(function_name="solve_it")
    assert "def solve_it(" in script


def test_ivp_ode_parse_error_message_is_user_facing():
    with pytest.raises(ValueError, match=r"Could not parse ODE equation"):
        IVPProblem.from_strings(
            equations=["y''(t) + * y(t) = 0"],
            initial_conditions=["y(t0) = 1", "y'(t0) = 0"],
        )


def test_ivp_ic_rhs_parse_error_message_is_user_facing():
    with pytest.raises(ValueError, match=r"Could not parse initial condition RHS"):
        IVPProblem.from_strings(
            equations=["y''(t) + y(t) = 0"],
            initial_conditions=["y(t0) = 1+", "y'(t0) = 0"],
        )
