import numpy as np

from ode_string_solver import (
    prepare_bvp_problem,
    solve_bvp_from_problem,
)


def test_bvp_prepare_with_implicit_bc_and_parameter():
    problem = prepare_bvp_problem(
        equations=["d2y/dx2 + q*y(x) = 0"],
        boundary_conditions=[
            "y(0) = 0",
            "y(1)^2 = 2",
            "y'(0) - q = 0",
        ],
        initial_guess=["x", "1"],
        left_boundary="0",
        right_boundary="1",
        parameter_names=["q"],
        parameter_guess=["10"],
    )

    assert len(problem.first_order_system.state_symbols) == 2
    assert len(problem.boundary_residuals) == 3
    assert [str(s) for s in problem.parameter_symbols] == ["q"]


def test_bvp_solve_smoke():
    problem = prepare_bvp_problem(
        equations=["d2y/dx2 + q*y(x) = 0"],
        boundary_conditions=[
            "y(0) = 0",
            "y(1)^2 = 2",
            "y'(0) - q = 0",
        ],
        initial_guess=["x", "1"],
        left_boundary="0",
        right_boundary="1",
        parameter_names=["q"],
        parameter_guess=["10"],
    )

    sol = solve_bvp_from_problem(problem, x_mesh=[i / 19 for i in range(20)], max_nodes=10000)

    assert sol.success
    assert sol.y.shape[0] == 2
    assert sol.p.shape[0] == 1


def test_bvp_solve_matches_closed_form_sine_solution():
    problem = prepare_bvp_problem(
        equations=["d2y/dx2 + y(x) = 0"],
        boundary_conditions=[
            "y(0) = 0",
            "y(1.5707963267948966) = 1",
        ],
        initial_guess=["x", "1"],
        left_boundary="0",
        right_boundary="1.5707963267948966",
    )

    x_mesh = np.linspace(0.0, np.pi / 2.0, 41)
    sol = solve_bvp_from_problem(problem, x_mesh=x_mesh, tol=1e-8, max_nodes=10000)

    assert sol.success
    np.testing.assert_allclose(sol.y[0], np.sin(sol.x), rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(sol.y[1], np.cos(sol.x), rtol=2e-5, atol=2e-6)