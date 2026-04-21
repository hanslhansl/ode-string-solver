from ode_string_solver import (
    generate_scipy_bvp_script,
    generate_scipy_ivp_script,
    prepare_bvp_problem,
    prepare_ivp_problem,
)


if __name__ == "__main__":
    ivp_problem = prepare_ivp_problem(
        equations=["d2 y / dt2 + y(t) = 0"],
        initial_conditions=["y(0) = 1", "y'(0) = 0"],
    )
    ivp_code = generate_scipy_ivp_script(ivp_problem, function_name="solve_ivp_from_generated_code")

    bvp_problem = prepare_bvp_problem(
        equations=["d2y/dx2 + y(x) = 0"],
        boundary_conditions=[
            "y(0) = 0",
            "y(1.5707963267948966) = 1",
        ],
        initial_guess=["x", "1"],
        left_boundary="0",
        right_boundary="1.5707963267948966",
    )
    bvp_code = generate_scipy_bvp_script(bvp_problem, function_name="solve_bvp_from_generated_code")

    print("=== Generated IVP script ===")
    print(ivp_code)
    print()
    print("=== Generated BVP script ===")
    print(bvp_code)
