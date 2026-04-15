from ode_string_solver import (
    build_ivp_callable,
    prepare_ivp_problem,
    solve_ivp_from_problem,
)


def test_ivp_prepare_and_callable_eval():
    problem = prepare_ivp_problem(
        equations=[
            "d2 y / dt2 + c*dy/dt + k*y(t) = 0",
            "z'(t) - y(t) = 0",
        ],
        initial_conditions=[
            "y(0) = 1",
            "y'(0) = 0",
            "z(0) = 2",
        ],
    )

    model = build_ivp_callable(problem)
    out = model.fun(0.0, [1.0, 0.0, 2.0], params={"c": 0.3, "k": 4.0})

    assert out.shape == (3,)
    assert out[0] == 0.0
    assert out[1] == -4.0
    assert out[2] == 1.0


def test_ivp_solve_smoke():
    problem = prepare_ivp_problem(
        equations=["d2 y / dt2 + c*dy/dt + k*y(t) = 0"],
        initial_conditions=["y(0) = 1", "y'(0) = 0"],
    )
    sol = solve_ivp_from_problem(
        problem,
        t_span=(0.0, 2.0),
        params={"c": 0.3, "k": 2.0},
    )

    assert sol.success
    assert sol.y.shape[0] == 2
