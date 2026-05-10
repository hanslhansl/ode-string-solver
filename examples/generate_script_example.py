from ode_string_solver import BVPProblem, IVPProblem


if __name__ == "__main__":
    ivp_problem = IVPProblem.from_strings(
        equations=["y''(t) + y(t) = 0"],
        initial_conditions=["y(t0) = 1", "y'(t0) = 0"],
    )
    ivp_code = ivp_problem.generate_scipy_script()

    bvp_problem = BVPProblem.from_strings(
        equations=["y''(x) + y(x) = 0"],
        boundary_conditions=[
            "y(a) = 0",
            "y(b) = 1",
        ],
        initial_guess=["x", "1"],
    )
    bvp_code = bvp_problem.generate_scipy_script()

    print("=== Generated IVP script ===")
    print(ivp_code)
    print()
    print("=== Generated BVP script ===")
    print(bvp_code)
