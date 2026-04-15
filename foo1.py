import numpy as np

from ode_string_solver import (
    prepare_bvp_problem,
    prepare_ivp_problem,
    solve_bvp_from_problem,
    solve_ivp_from_problem,
)


if __name__ == "__main__":
    print("Legacy demo script. Prefer package imports from 'ode_string_solver'.")

    ivp_problem = prepare_ivp_problem(
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
    ivp_solution = solve_ivp_from_problem(
        ivp_problem,
        t_span=(0.0, 6.0),
        t_eval=np.linspace(0.0, 6.0, 40),
        params={"c": 0.3, "k": 4.0},
    )
    print("solve_ivp success:", ivp_solution.success)

    bvp_problem = prepare_bvp_problem(
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
    bvp_solution = solve_bvp_from_problem(
        bvp_problem,
        x_mesh=np.linspace(0.0, 1.0, 20),
        max_nodes=10000,
    )
    print("solve_bvp success:", bvp_solution.success)
