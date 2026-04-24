from ode_string_solver import BVPProblem, IVPProblem


if __name__ == "__main__":
    ivp_problem = IVPProblem.from_strings(
        equations=["d2 y / dt2 + y(t) = 0"],
        initial_conditions=["y(0) = 1", "y'(0) = 0"],
    )
    ivp_code = ivp_problem.generate_scipy_script(function_name="solve_ivp_from_generated_code")

    bvp_problem = BVPProblem.from_strings(
        equations=["d2y/dx2 + y(x) = 0"],
        boundary_conditions=[
            "y(0) = 0",
            "y(1.5707963267948966) = 1",
        ],
        initial_guess=["x", "1"],
        left_boundary="0",
        right_boundary="1.5707963267948966",
    )
    bvp_code = bvp_problem.generate_scipy_script(function_name="solve_bvp_from_generated_code")

    print("=== Generated IVP script ===")
    print(ivp_code)
    print()
    print("=== Generated BVP script ===")
    print(bvp_code)
