import numpy as np

from ode_string_solver import prepare_ivp_problem, solve_ivp_from_problem


if __name__ == "__main__":
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

    sol = solve_ivp_from_problem(
        problem,
        t_span=(0.0, 6.0),
        t_eval=np.linspace(0.0, 6.0, 40),
        params={"c": 0.3, "k": 4.0},
    )

    print("success:", sol.success)
    print("final state:", sol.y[:, -1])
