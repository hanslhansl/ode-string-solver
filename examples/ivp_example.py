import numpy as np

from ode_string_solver import IVPProblem


if __name__ == "__main__":
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

    sol = problem.solve(
        t_span=(0.0, 6.0),
        t_eval=np.linspace(0.0, 6.0, 40),
        namespace={"c": 0.3, "k": 4.0},
    )

    print("success:", sol.success)
    print("final state:", sol.y[:, -1])
