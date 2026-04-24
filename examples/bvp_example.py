import numpy as np

from ode_string_solver import BVPProblem


if __name__ == "__main__":
    problem = BVPProblem.from_strings(
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

    sol = problem.solve(x_mesh=np.linspace(0.0, 1.0, 20), max_nodes=10000)

    print("success:", sol.success)
    print("estimated q:", sol.p)
