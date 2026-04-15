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
