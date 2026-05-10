import numpy as np

from ode_string_solver import BVPProblem


def test_bvp_prepare_with_implicit_bc_and_parameter():
    problem = BVPProblem.from_strings(
        equations=["y''(x) + q*y(x) = 0"],
        boundary_conditions=[
            "y(a) = 0",
            "y(b)**2 = 2",
            "y'(a) - q = 0",
        ],
        initial_guess=["x", "1"],
        parameter_names=["q"],
        parameter_guess=["10"],
    )

    assert len(problem.first_order_system.state_symbols) == 2
    assert len(problem.boundary_residuals) == 3
    assert [str(s) for s in problem.parameter_symbols] == ["q"]


def test_bvp_solve_smoke():
    problem = BVPProblem.from_strings(
        equations=["y''(x) + q*y(x) = 0"],
        boundary_conditions=[
            "y(a) = 0",
            "y(b)**2 = 2",
            "y'(a) - q = 0",
        ],
        initial_guess=["x", "1"],
        parameter_names=["q"],
        parameter_guess=["10"],
    )

    sol = problem.solve(x_mesh=[i / 19 for i in range(20)], max_nodes=10000)

    assert sol.success
    assert sol.y.shape[0] == 2
    assert sol.p.shape[0] == 1


def test_bvp_solve_matches_closed_form_sine_solution():
    problem = BVPProblem.from_strings(
        equations=["y''(x) + y(x) = 0"],
        boundary_conditions=[
            "y(a) = 0",
            "y(b) = 1",
        ],
        initial_guess=["x", "1"],
    )

    x_mesh = np.linspace(0.0, np.pi / 2.0, 41)
    sol = problem.solve(x_mesh=x_mesh, tol=1e-8, max_nodes=10000)

    assert sol.success
    np.testing.assert_allclose(sol.y[0], np.sin(sol.x), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(sol.y[1], np.cos(sol.x), rtol=2e-5, atol=2e-6)


def test_bvp_method_api_smoke():
    problem = BVPProblem.from_strings(
        equations=["y''(x) + y(x) = 0"],
        boundary_conditions=[
            "y(a) = 0",
            "y(b) = 1",
        ],
        initial_guess=["x", "1"],
    )

    model = problem.build_callables()
    assert callable(model.fun)
    assert callable(model.bc)

    x_mesh = np.linspace(0.0, np.pi / 2.0, 41)
    sol = problem.solve(x_mesh=x_mesh, tol=1e-8, max_nodes=10000)
    assert sol.success

    script = problem.generate_scipy_script(function_name="solve_it")
    assert "def solve_it(" in script


def test_bvp_callable_parameter_and_builtin_math_function_work_together():
    problem = BVPProblem.from_strings(
        equations=["y'(x) - a(c, d)*cos(x) = 0"],
        boundary_conditions=[
            "y(x0) = 0",
        ],
        initial_guess=["0"],
        left_symbol="x0",
        right_symbol="x1",
    )

    sol = problem.solve(
        x_mesh=np.linspace(0.0, 1.0, 21),
        namespace={"a": lambda x, y: 1.0, "cos": np.cos, "sin": np.sin, "c" : 12., "d" : 13.},
        tol=1e-8,
        max_nodes=10000,
    )

    assert sol.success
    np.testing.assert_allclose(sol.y[0], np.sin(sol.x), rtol=2e-5, atol=2e-6)