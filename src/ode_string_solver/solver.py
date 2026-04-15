from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.printing.numpy import NumPyPrinter


TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)
IDENTIFIER = r"[A-Za-z_]\w*"


@dataclass(frozen=True)
class ParsedDifferentialSystem:
    """Result of parsing user equation strings into SymPy objects."""

    normalized_equations: list[str]
    equations: list[sp.Equality]
    independent_variable: str
    target_functions: list[str]
    highest_derivatives: list[sp.Derivative]
    solved_highest_derivatives: list[sp.Expr]


@dataclass(frozen=True)
class FirstOrderSystem:
    """First-order representation x' = f(t, x) derived from ODE input."""

    independent_variable: sp.Symbol
    state_symbols: list[sp.Symbol]
    state_descriptions: list[str]
    rhs_expressions: list[sp.Expr]
    max_orders: dict[str, int]


@dataclass(frozen=True)
class IVPProblem:
    """Parsed IVP input bound to first-order state order."""

    parsed_system: ParsedDifferentialSystem
    first_order_system: FirstOrderSystem
    t0: sp.Expr
    y0: list[sp.Expr]


@dataclass(frozen=True)
class BVPProblem:
    """Parsed BVP input bound to first-order state order."""

    parsed_system: ParsedDifferentialSystem
    first_order_system: FirstOrderSystem
    left_boundary: sp.Expr
    right_boundary: sp.Expr
    left_state_symbols: list[sp.Symbol]
    right_state_symbols: list[sp.Symbol]
    boundary_residuals: list[sp.Expr]
    initial_guess_expressions: list[sp.Expr]
    parameter_symbols: list[sp.Symbol]
    parameter_guess: list[sp.Expr]


@dataclass(frozen=True)
class IVPCallableModel:
    """Numeric callable model for solve_ivp."""

    fun: Callable[..., np.ndarray]
    parameter_symbols: list[sp.Symbol]


@dataclass(frozen=True)
class BVPCallableModel:
    """Numeric callable model for solve_bvp."""

    fun: Callable[..., np.ndarray]
    bc: Callable[..., np.ndarray]
    fixed_parameter_symbols: list[sp.Symbol]
    unknown_parameter_symbols: list[sp.Symbol]


def normalize_differential_notation(text: str) -> str:
    """Convert common derivative notations to SymPy Derivative(...)."""

    normalized = text

    def replace_prime(match: re.Match[str]) -> str:
        function_name = match.group("fn")
        prime_marks = match.group("pr")
        variable = match.group("var")
        order = len(prime_marks)
        return f"Derivative({function_name}({variable}), {variable}, {order})"

    normalized = re.sub(
        rf"\b(?P<fn>{IDENTIFIER})\s*(?P<pr>'{{1,6}})\s*\(\s*(?P<var>{IDENTIFIER})\s*\)",
        replace_prime,
        normalized,
    )

    def replace_leibniz_compact(match: re.Match[str]) -> str:
        order_token = match.group("ord")
        function_name = match.group("fn")
        variable = match.group("var")
        order = int(order_token) if order_token else 1
        if order == 1:
            return f"Derivative({function_name}({variable}), {variable})"
        return f"Derivative({function_name}({variable}), {variable}, {order})"

    normalized = re.sub(
        rf"d(?P<ord>\d*)\s*(?P<fn>{IDENTIFIER})\s*/\s*d(?P<var>{IDENTIFIER})(?P=ord)",
        replace_leibniz_compact,
        normalized,
    )

    normalized = re.sub(
        rf"d\s*/\s*d(?P<var>{IDENTIFIER})\s*(?P<fn>{IDENTIFIER})\s*\(\s*(?P=var)\s*\)",
        r"Derivative(\g<fn>(\g<var>), \g<var>)",
        normalized,
    )

    return normalized


def normalize_boundary_condition_notation(text: str, independent_variable: sp.Symbol) -> str:
    """Normalize BC prime notation like y'(1) to SymPy Derivative expressions."""

    independent_name = str(independent_variable)
    normalized = text

    def replace_prime_with_point(match: re.Match[str]) -> str:
        function_name = match.group("fn")
        prime_marks = match.group("pr")
        point = match.group("point")
        order = len(prime_marks)
        return (
            f"Subs(Derivative({function_name}({independent_name}), {independent_name}, {order}), "
            f"{independent_name}, ({point}))"
        )

    normalized = re.sub(
        rf"\b(?P<fn>{IDENTIFIER})\s*(?P<pr>'{{1,6}})\s*\(\s*(?P<point>[^()]+)\s*\)",
        replace_prime_with_point,
        normalized,
    )

    return normalized


def _build_local_dict(
    normalized_equations: Sequence[str],
    namespace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    local_dict: dict[str, Any] = {
        "Derivative": sp.Derivative,
        "Eq": sp.Eq,
    }
    if namespace is not None:
        local_dict.update(namespace)
    combined = " ; ".join(normalized_equations)

    called_names = set(re.findall(rf"\b({IDENTIFIER})\s*\(", combined))
    identifier_names = set(re.findall(rf"\b{IDENTIFIER}\b", combined))

    for name in sorted(identifier_names):
        if name in local_dict:
            continue
        if name in called_names:
            local_dict[name] = sp.Function(name)
        else:
            local_dict[name] = sp.Symbol(name)

    return local_dict


def _parse_equation(equation: str, local_dict: dict[str, Any]) -> sp.Equality:
    if "=" in equation:
        lhs_raw, rhs_raw = equation.split("=", 1)
    else:
        lhs_raw, rhs_raw = equation, "0"

    lhs = parse_expr(
        lhs_raw,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=False,
    )
    rhs = parse_expr(
        rhs_raw,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=False,
    )
    return cast(sp.Equality, sp.Eq(lhs, rhs, evaluate=False))


def _extract_independent_variable(equations: Sequence[sp.Equality]) -> sp.Symbol:
    variables: set[sp.Symbol] = set()
    for equation in equations:
        for derivative in equation.atoms(sp.Derivative):
            for variable in derivative.variables:
                variables.add(variable)

    if len(variables) != 1:
        variable_names = sorted(str(variable) for variable in variables)
        raise ValueError(
            "Expected exactly one independent variable across all equations; "
            f"got {variable_names}."
        )
    return next(iter(variables))


def _extract_target_functions(equations: Sequence[sp.Equality]) -> list[str]:
    targets: set[str] = set()
    for equation in equations:
        for derivative in equation.atoms(sp.Derivative):
            if isinstance(derivative.expr, sp.Function):
                targets.add(str(derivative.expr.func))
            else:
                raise ValueError(
                    "Derivative target is not a function application: "
                    f"{sp.srepr(derivative)}"
                )

    if not targets:
        raise ValueError("No derivatives found. At least one ODE is required.")
    return sorted(targets)


def _highest_derivative_in_equation(equation: sp.Equality) -> sp.Derivative:
    derivatives = sorted(
        equation.atoms(sp.Derivative),
        key=lambda item: len(item.variables),
        reverse=True,
    )
    if not derivatives:
        raise ValueError(f"No derivative found in equation: {equation}")

    highest_order = len(derivatives[0].variables)
    highest = [d for d in derivatives if len(d.variables) == highest_order]
    if len(highest) != 1:
        raise ValueError(
            "Ambiguous highest derivative in equation. "
            f"Found: {[str(item) for item in highest]}"
        )
    return highest[0]


def _build_state_lookup(first_order: FirstOrderSystem) -> dict[tuple[str, int], sp.Symbol]:
    lookup: dict[tuple[str, int], sp.Symbol] = {}
    for state_symbol in first_order.state_symbols:
        state_name = str(state_symbol)
        function_name, order_token = state_name.rsplit("_", 1)
        lookup[(function_name, int(order_token))] = state_symbol
    return lookup


def _parse_function_order_lhs(lhs: str) -> tuple[str, int, str]:
    pattern = rf"^\s*(?P<fn>{IDENTIFIER})\s*(?P<pr>'{{0,6}})\s*\(\s*(?P<point>[^()]+)\s*\)\s*$"
    match = re.match(pattern, lhs)
    if not match:
        raise ValueError(
            "Condition LHS must look like f(t0), f'(t0), f''(t0), ... . "
            f"Got '{lhs}'."
        )
    function_name = match.group("fn")
    order = len(match.group("pr"))
    point_text = match.group("point")
    return function_name, order, point_text


def _parse_scalar_expr(expr_text: str, namespace: Mapping[str, Any] | None) -> sp.Expr:
    local_dict: dict[str, Any] = {}
    if namespace is not None:
        local_dict.update(namespace)
    expr = parse_expr(
        expr_text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=False,
    )
    return cast(sp.Expr, expr)


def _parse_condition_equation(condition: str) -> tuple[str, str]:
    if "=" not in condition:
        raise ValueError(f"Condition must contain '='. Got '{condition}'.")
    lhs, rhs = condition.split("=", 1)
    return lhs.strip(), rhs.strip()


def _parse_residual_equation(equation: str, local_dict: dict[str, Any]) -> sp.Expr:
    if "=" in equation:
        lhs_raw, rhs_raw = equation.split("=", 1)
        lhs_expr = parse_expr(
            lhs_raw,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=False,
        )
        rhs_expr = parse_expr(
            rhs_raw,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=False,
        )
        return _expr_diff(cast(sp.Expr, lhs_expr), cast(sp.Expr, rhs_expr))

    expr = parse_expr(
        equation,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=False,
    )
    return cast(sp.Expr, expr)


def _expr_diff(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    return cast(
        sp.Expr,
        sp.Add(left, sp.Mul(sp.Integer(-1), right, evaluate=False), evaluate=False),
    )


def parse_differential_system(
    equations: Sequence[str],
    namespace: Mapping[str, Any] | None = None,
) -> ParsedDifferentialSystem:
    """Parse ODE strings and solve each equation for its highest derivative."""

    if not equations:
        raise ValueError("At least one equation string must be provided.")

    normalized_equations = [normalize_differential_notation(eq) for eq in equations]
    local_dict = _build_local_dict(normalized_equations, namespace)
    parsed_equations = [_parse_equation(eq, local_dict) for eq in normalized_equations]

    independent_variable = _extract_independent_variable(parsed_equations)
    target_functions = _extract_target_functions(parsed_equations)

    highest_derivatives = [_highest_derivative_in_equation(eq) for eq in parsed_equations]
    solved_highest_derivatives: list[sp.Expr] = []

    for equation, highest in zip(parsed_equations, highest_derivatives):
        solutions = sp.solve(equation, highest, dict=True)
        if not solutions:
            raise ValueError(
                f"Could not solve equation '{equation}' for highest derivative '{highest}'."
            )
        solved_highest_derivatives.append(solutions[0][highest])

    return ParsedDifferentialSystem(
        normalized_equations=normalized_equations,
        equations=parsed_equations,
        independent_variable=str(independent_variable),
        target_functions=target_functions,
        highest_derivatives=highest_derivatives,
        solved_highest_derivatives=solved_highest_derivatives,
    )


def prepare_ivp_problem(
    equations: Sequence[str],
    initial_conditions: Sequence[str],
    namespace: Mapping[str, Any] | None = None,
) -> IVPProblem:
    """Parse ODE+IC strings into first-order state initial values."""

    parsed = parse_differential_system(equations, namespace=namespace)
    first_order = decouple_to_first_order(parsed)
    state_lookup = _build_state_lookup(first_order)

    if len(initial_conditions) != len(first_order.state_symbols):
        raise ValueError(
            "Initial condition count must match state count. "
            f"Expected {len(first_order.state_symbols)}, got {len(initial_conditions)}."
        )

    initial_values_by_key: dict[tuple[str, int], sp.Expr] = {}
    t0_value: sp.Expr | None = None

    for condition in initial_conditions:
        lhs, rhs = _parse_condition_equation(condition)
        function_name, order, point_text = _parse_function_order_lhs(lhs)
        point_expr = _parse_scalar_expr(point_text, namespace)
        rhs_expr = _parse_scalar_expr(rhs, namespace)
        key = (function_name, order)

        if key not in state_lookup:
            raise ValueError(
                "Initial condition refers to unknown state term "
                f"'{function_name}' order {order}."
            )

        if key in initial_values_by_key:
            raise ValueError(f"Duplicate initial condition for '{function_name}' order {order}.")

        if t0_value is None:
            t0_value = point_expr
        elif sp.simplify(_expr_diff(point_expr, t0_value)) != 0:
            raise ValueError(
                "All initial conditions must use the same initial point. "
                f"Found {t0_value} and {point_expr}."
            )

        initial_values_by_key[key] = rhs_expr

    missing = [key for key in state_lookup if key not in initial_values_by_key]
    if missing:
        missing_text = [f"{name}^{order}" for name, order in missing]
        raise ValueError(f"Missing initial conditions for {missing_text}.")

    y0: list[sp.Expr] = []
    for state_symbol in first_order.state_symbols:
        function_name, order_token = str(state_symbol).rsplit("_", 1)
        y0.append(initial_values_by_key[(function_name, int(order_token))])

    if t0_value is None:
        raise ValueError("No initial conditions provided.")

    return IVPProblem(
        parsed_system=parsed,
        first_order_system=first_order,
        t0=t0_value,
        y0=y0,
    )


def _parse_bvp_initial_guess(
    first_order: FirstOrderSystem,
    initial_guess: Sequence[str] | Mapping[str, str],
    namespace: Mapping[str, Any] | None,
) -> list[sp.Expr]:
    if isinstance(initial_guess, Mapping):
        guess_map: dict[str, sp.Expr] = {}
        for key, value in initial_guess.items():
            guess_map[key] = _parse_scalar_expr(value, namespace)

        missing = [str(symbol) for symbol in first_order.state_symbols if str(symbol) not in guess_map]
        if missing:
            raise ValueError(
                "Missing BVP initial guess expressions for states "
                f"{missing}."
            )
        return [guess_map[str(symbol)] for symbol in first_order.state_symbols]

    if len(initial_guess) != len(first_order.state_symbols):
        raise ValueError(
            "BVP initial guess expression count must match state count. "
            f"Expected {len(first_order.state_symbols)}, got {len(initial_guess)}."
        )

    return [_parse_scalar_expr(expr, namespace) for expr in initial_guess]


def prepare_bvp_problem(
    equations: Sequence[str],
    boundary_conditions: Sequence[str],
    initial_guess: Sequence[str] | Mapping[str, str],
    left_boundary: str,
    right_boundary: str,
    parameter_names: Sequence[str] | None = None,
    parameter_guess: Sequence[str] | None = None,
    namespace: Mapping[str, Any] | None = None,
) -> BVPProblem:
    """Parse ODE+BVP strings into first-order residual form for solve_bvp."""

    parsed = parse_differential_system(equations, namespace=namespace)
    first_order = decouple_to_first_order(parsed)
    state_lookup = _build_state_lookup(first_order)
    independent_variable = first_order.independent_variable

    left_expr = _parse_scalar_expr(left_boundary, namespace)
    right_expr = _parse_scalar_expr(right_boundary, namespace)
    if sp.simplify(_expr_diff(left_expr, right_expr)) == 0:
        raise ValueError("Left and right boundaries must be different.")

    left_state_symbols = [sp.Symbol(f"{symbol}_left") for symbol in first_order.state_symbols]
    right_state_symbols = [sp.Symbol(f"{symbol}_right") for symbol in first_order.state_symbols]

    left_subs: dict[sp.Expr, sp.Symbol] = {}
    right_subs: dict[sp.Expr, sp.Symbol] = {}
    for state_symbol in first_order.state_symbols:
        function_name, order_token = str(state_symbol).rsplit("_", 1)
        order = int(order_token)
        function_obj = sp.Function(function_name)
        base = cast(sp.Expr, function_obj(independent_variable))

        if order == 0:
            left_term = cast(sp.Expr, function_obj(left_expr))
            right_term = cast(sp.Expr, function_obj(right_expr))
        else:
            left_term = cast(
                sp.Expr,
                sp.Subs(
                    sp.Derivative(base, independent_variable, order),
                    independent_variable,
                    left_expr,
                ),
            )
            right_term = cast(
                sp.Expr,
                sp.Subs(
                    sp.Derivative(base, independent_variable, order),
                    independent_variable,
                    right_expr,
                ),
            )

        left_subs[left_term] = left_state_symbols[len(left_subs)]
        right_subs[right_term] = right_state_symbols[len(right_subs)]

    param_symbols = [sp.Symbol(name) for name in (parameter_names or [])]
    parameter_guess_exprs: list[sp.Expr] = []
    if parameter_names is not None and parameter_guess is not None:
        if len(parameter_names) != len(parameter_guess):
            raise ValueError("parameter_names and parameter_guess must have the same length.")
        parameter_guess_exprs = [_parse_scalar_expr(expr, namespace) for expr in parameter_guess]
    elif parameter_names is not None and parameter_guess is None:
        raise ValueError("parameter_guess is required when parameter_names are provided.")
    elif parameter_names is None and parameter_guess is not None:
        raise ValueError("parameter_names is required when parameter_guess is provided.")

    bc_local_dict = _build_local_dict(boundary_conditions, namespace)
    bc_local_dict[str(independent_variable)] = independent_variable
    for function_name in parsed.target_functions:
        bc_local_dict[function_name] = sp.Function(function_name)

    boundary_residuals: list[sp.Expr] = []
    for condition in boundary_conditions:
        normalized_condition = normalize_boundary_condition_notation(condition, independent_variable)
        residual = _parse_residual_equation(normalized_condition, bc_local_dict)
        residual = cast(sp.Expr, residual.subs(left_subs))
        residual = cast(sp.Expr, residual.subs(right_subs))

        target_names = set(parsed.target_functions)
        unresolved_functions = [
            f for f in residual.atoms(sp.Function) if str(f.func) in target_names
        ]
        unresolved_derivatives = [
            d for d in residual.atoms(sp.Derivative) if str(d.expr.func) in target_names
        ]
        if unresolved_functions or unresolved_derivatives:
            raise ValueError(
                "Boundary condition must only reference target function values/derivatives "
                "at the left or right boundary. Unresolved terms remain in "
                f"'{condition}'."
            )

        boundary_residuals.append(residual)

    expected_residual_count = len(first_order.state_symbols) + len(param_symbols)
    if len(boundary_residuals) != expected_residual_count:
        raise ValueError(
            "Boundary condition count must be state_count + parameter_count for solve_bvp. "
            f"Expected {expected_residual_count}, got {len(boundary_residuals)}."
        )

    guess_expressions = _parse_bvp_initial_guess(first_order, initial_guess, namespace)

    return BVPProblem(
        parsed_system=parsed,
        first_order_system=first_order,
        left_boundary=left_expr,
        right_boundary=right_expr,
        left_state_symbols=left_state_symbols,
        right_state_symbols=right_state_symbols,
        boundary_residuals=boundary_residuals,
        initial_guess_expressions=guess_expressions,
        parameter_symbols=param_symbols,
        parameter_guess=parameter_guess_exprs,
    )


def decouple_to_first_order(system: ParsedDifferentialSystem) -> FirstOrderSystem:
    """Convert parsed higher-order ODEs into a first-order state-space system."""

    independent_variable = sp.Symbol(system.independent_variable)
    candidates: dict[str, list[tuple[int, sp.Derivative, sp.Expr]]] = {}

    for highest, rhs in zip(system.highest_derivatives, system.solved_highest_derivatives):
        if not isinstance(highest.expr, sp.Function):
            raise ValueError(f"Invalid highest derivative target: {highest}")
        if any(variable != independent_variable for variable in highest.variables):
            raise ValueError(
                "Highest derivative uses an unexpected variable in "
                f"{highest}. Expected only {independent_variable}."
            )
        if highest.expr.args != (independent_variable,):
            raise ValueError(
                "Target functions must be single-variable applications like f(t). "
                f"Got {highest.expr}."
            )

        function_name = str(highest.expr.func)
        order = len(highest.variables)
        candidates.setdefault(function_name, []).append((order, highest, rhs))

    representative: dict[str, tuple[int, sp.Derivative, sp.Expr]] = {}
    max_orders: dict[str, int] = {}
    for function_name in system.target_functions:
        if function_name not in candidates:
            raise ValueError(f"No equation found for target function '{function_name}'.")
        items = candidates[function_name]
        max_order = max(order for order, _, _ in items)
        best_items = [item for item in items if item[0] == max_order]
        if len(best_items) != 1:
            raise ValueError(
                "Ambiguous highest-order equation for function "
                f"'{function_name}' at order {max_order}."
            )
        representative[function_name] = best_items[0]
        max_orders[function_name] = max_order

    substitutions: dict[sp.Expr, sp.Expr] = {}
    state_symbols: list[sp.Symbol] = []
    state_descriptions: list[str] = []
    state_lookup: dict[tuple[str, int], sp.Symbol] = {}

    for function_name in system.target_functions:
        _, highest, _ = representative[function_name]
        base = cast(sp.Function, highest.expr)
        for order in range(max_orders[function_name]):
            state_symbol = sp.Symbol(f"{function_name}_{order}")
            state_symbols.append(state_symbol)
            state_descriptions.append(f"{function_name}^{order}")
            state_lookup[(function_name, order)] = state_symbol

            if order == 0:
                substitutions[base] = state_symbol
            else:
                substitutions[sp.Derivative(base, independent_variable, order)] = state_symbol

    rhs_expressions: list[sp.Expr] = []
    for function_name in system.target_functions:
        highest_order, _, highest_rhs = representative[function_name]

        for order in range(highest_order - 1):
            rhs_expressions.append(state_lookup[(function_name, order + 1)])

        substituted_highest_rhs = cast(sp.Expr, highest_rhs.subs(substitutions))
        remaining_derivatives = substituted_highest_rhs.atoms(sp.Derivative)
        if remaining_derivatives:
            raise ValueError(
                "Could not fully decouple system. Unresolved derivatives remain in "
                f"RHS for '{function_name}': {[str(item) for item in remaining_derivatives]}"
            )

        rhs_expressions.append(substituted_highest_rhs)

    return FirstOrderSystem(
        independent_variable=independent_variable,
        state_symbols=state_symbols,
        state_descriptions=state_descriptions,
        rhs_expressions=rhs_expressions,
        max_orders=max_orders,
    )


def _ordered_parameter_symbols(
    expressions: Sequence[sp.Expr],
    excluded_symbols: set[sp.Symbol],
) -> list[sp.Symbol]:
    parameter_set: set[sp.Symbol] = set()
    for expr in expressions:
        for symbol in expr.free_symbols:
            if isinstance(symbol, sp.Symbol):
                parameter_set.add(symbol)
    parameter_set -= excluded_symbols
    return sorted(parameter_set, key=lambda symbol: str(symbol))


def _resolve_parameter_values(
    parameter_symbols: Sequence[sp.Symbol],
    params: Mapping[str, float] | Sequence[float] | None,
) -> list[float]:
    if not parameter_symbols:
        return []

    if params is None:
        names = [str(symbol) for symbol in parameter_symbols]
        raise ValueError(f"Missing parameter values for {names}.")

    if isinstance(params, Mapping):
        values: list[float] = []
        for symbol in parameter_symbols:
            name = str(symbol)
            if name not in params:
                raise ValueError(f"Missing parameter value for '{name}'.")
            values.append(float(params[name]))
        return values

    if len(params) != len(parameter_symbols):
        raise ValueError(
            "Parameter sequence length mismatch. "
            f"Expected {len(parameter_symbols)}, got {len(params)}."
        )
    return [float(value) for value in params]


def build_ivp_callable(problem: IVPProblem) -> IVPCallableModel:
    """Build a numeric RHS callable compatible with scipy.integrate.solve_ivp."""

    independent = problem.first_order_system.independent_variable
    state_symbols = problem.first_order_system.state_symbols
    expressions = problem.first_order_system.rhs_expressions

    excluded = {independent, *state_symbols}
    parameter_symbols = _ordered_parameter_symbols(expressions, excluded)

    args = [independent, *state_symbols, *parameter_symbols]
    rhs_lambda = sp.lambdify(args, expressions, modules="numpy")
    state_count = len(state_symbols)

    def fun(
        t: float,
        y: Sequence[float] | np.ndarray,
        params: Mapping[str, float] | Sequence[float] | None = None,
    ) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if y_arr.shape[0] != state_count:
            raise ValueError(
                "State vector length mismatch. "
                f"Expected {state_count}, got {y_arr.shape[0]}."
            )

        parameter_values = _resolve_parameter_values(parameter_symbols, params)
        values = rhs_lambda(float(t), *y_arr.tolist(), *parameter_values)
        return np.asarray(values, dtype=float).reshape(state_count)

    return IVPCallableModel(fun=fun, parameter_symbols=parameter_symbols)


def build_bvp_callables(problem: BVPProblem) -> BVPCallableModel:
    """Build numeric fun/bc callables compatible with scipy.integrate.solve_bvp."""

    independent = problem.first_order_system.independent_variable
    state_symbols = problem.first_order_system.state_symbols
    unknown_parameter_symbols = problem.parameter_symbols
    rhs_expressions = problem.first_order_system.rhs_expressions
    bc_expressions = problem.boundary_residuals

    fun_excluded = {independent, *state_symbols, *unknown_parameter_symbols}
    bc_excluded = {
        *problem.left_state_symbols,
        *problem.right_state_symbols,
        *unknown_parameter_symbols,
    }
    fixed_fun_parameters = _ordered_parameter_symbols(rhs_expressions, fun_excluded)
    fixed_bc_parameters = _ordered_parameter_symbols(bc_expressions, bc_excluded)
    fixed_parameter_symbols = sorted(
        set(fixed_fun_parameters) | set(fixed_bc_parameters),
        key=lambda symbol: str(symbol),
    )

    fun_args = [independent, *state_symbols, *unknown_parameter_symbols, *fixed_parameter_symbols]
    bc_args = [
        *problem.left_state_symbols,
        *problem.right_state_symbols,
        *unknown_parameter_symbols,
        *fixed_parameter_symbols,
    ]

    fun_lambda = sp.lambdify(fun_args, rhs_expressions, modules="numpy")
    bc_lambda = sp.lambdify(bc_args, bc_expressions, modules="numpy")
    state_count = len(state_symbols)
    unknown_count = len(unknown_parameter_symbols)

    def fun(
        x: Sequence[float] | np.ndarray,
        y: np.ndarray,
        p: Sequence[float] | np.ndarray,
        params: Mapping[str, float] | Sequence[float] | None = None,
    ) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim != 2 or y_arr.shape[0] != state_count:
            raise ValueError(
                "BVP state array must have shape (n, m). "
                f"Expected first dimension {state_count}, got {y_arr.shape}."
            )

        p_arr = np.asarray(p, dtype=float).reshape(-1)
        if p_arr.shape[0] != unknown_count:
            raise ValueError(
                "Unknown parameter vector length mismatch. "
                f"Expected {unknown_count}, got {p_arr.shape[0]}."
            )

        fixed_values = _resolve_parameter_values(fixed_parameter_symbols, params)
        x_arr = np.asarray(x, dtype=float)
        values = fun_lambda(
            x_arr,
            *[y_arr[idx] for idx in range(state_count)],
            *p_arr.tolist(),
            *fixed_values,
        )
        return np.asarray(values, dtype=float).reshape(state_count, -1)

    def bc(
        ya: Sequence[float] | np.ndarray,
        yb: Sequence[float] | np.ndarray,
        p: Sequence[float] | np.ndarray,
        params: Mapping[str, float] | Sequence[float] | None = None,
    ) -> np.ndarray:
        ya_arr = np.asarray(ya, dtype=float).reshape(-1)
        yb_arr = np.asarray(yb, dtype=float).reshape(-1)
        if ya_arr.shape[0] != state_count or yb_arr.shape[0] != state_count:
            raise ValueError(
                "Boundary state vector length mismatch. "
                f"Expected {state_count}, got ya={ya_arr.shape[0]}, yb={yb_arr.shape[0]}."
            )

        p_arr = np.asarray(p, dtype=float).reshape(-1)
        if p_arr.shape[0] != unknown_count:
            raise ValueError(
                "Unknown parameter vector length mismatch. "
                f"Expected {unknown_count}, got {p_arr.shape[0]}."
            )

        fixed_values = _resolve_parameter_values(fixed_parameter_symbols, params)
        values = bc_lambda(
            *ya_arr.tolist(),
            *yb_arr.tolist(),
            *p_arr.tolist(),
            *fixed_values,
        )
        return np.asarray(values, dtype=float).reshape(-1)

    return BVPCallableModel(
        fun=fun,
        bc=bc,
        fixed_parameter_symbols=fixed_parameter_symbols,
        unknown_parameter_symbols=unknown_parameter_symbols,
    )


def solve_ivp_from_problem(
    problem: IVPProblem,
    t_span: tuple[float, float],
    t_eval: Sequence[float] | np.ndarray | None = None,
    params: Mapping[str, float] | Sequence[float] | None = None,
    **solve_kwargs: Any,
):
    """Solve an IVPProblem via scipy.integrate.solve_ivp."""

    model = build_ivp_callable(problem)
    y0_numeric = np.asarray([float(sp.N(value)) for value in problem.y0], dtype=float)

    def wrapped_fun(t: float, y: np.ndarray) -> np.ndarray:
        return model.fun(t, y, params=params)

    return solve_ivp(wrapped_fun, t_span=t_span, y0=y0_numeric, t_eval=t_eval, **solve_kwargs)


def solve_bvp_from_problem(
    problem: BVPProblem,
    x_mesh: Sequence[float] | np.ndarray,
    params: Mapping[str, float] | Sequence[float] | None = None,
    **solve_kwargs: Any,
):
    """Solve a BVPProblem via scipy.integrate.solve_bvp."""

    model = build_bvp_callables(problem)
    x_arr = np.asarray(x_mesh, dtype=float)
    if x_arr.ndim != 1:
        raise ValueError("x_mesh must be a one-dimensional array.")

    initial_guess_lambda = sp.lambdify(
        [problem.first_order_system.independent_variable],
        problem.initial_guess_expressions,
        modules="numpy",
    )
    raw_guess = initial_guess_lambda(x_arr)
    y_rows: list[np.ndarray] = []
    for item in raw_guess:
        arr = np.asarray(item, dtype=float)
        if arr.ndim == 0:
            arr = np.full_like(x_arr, float(arr), dtype=float)
        else:
            arr = arr.reshape(-1)
            if arr.shape[0] != x_arr.shape[0]:
                raise ValueError(
                    "Initial guess expression produced wrong length. "
                    f"Expected {x_arr.shape[0]}, got {arr.shape[0]}."
                )
        y_rows.append(arr)
    y_guess = np.vstack(y_rows)

    p_guess = np.asarray([float(sp.N(value)) for value in problem.parameter_guess], dtype=float)

    def wrapped_fun(x: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return model.fun(x, y, p, params=params)

    def wrapped_bc(ya: np.ndarray, yb: np.ndarray, p: np.ndarray) -> np.ndarray:
        return model.bc(ya, yb, p, params=params)

    return solve_bvp(wrapped_fun, wrapped_bc, x_arr, y_guess, p=p_guess, **solve_kwargs)


def _expr_to_numpy_code(expr: sp.Expr) -> str:
    printer = NumPyPrinter()
    return cast(str, printer.doprint(expr))


def _state_symbol_to_function_order(state_symbol: sp.Symbol) -> tuple[str, int]:
    name = str(state_symbol)
    function_name, order_token = name.rsplit("_", 1)
    return function_name, int(order_token)


def _function_derivative_text(function_name: str, derivative_order: int, independent: sp.Symbol) -> str:
    primes = "'" * derivative_order
    return f"{function_name}{primes}({independent})"


def _function_derivative_at_point_text(
    function_name: str,
    derivative_order: int,
    point_expr: sp.Expr,
) -> str:
    primes = "'" * derivative_order
    return f"{function_name}{primes}({sp.sstr(point_expr)})"


def generate_scipy_ivp_script(problem: IVPProblem, function_name: str = "solve_problem") -> str:
    """Generate a standalone SciPy IVP solver script as a Python string."""

    state_symbols = problem.first_order_system.state_symbols
    independent = problem.first_order_system.independent_variable
    rhs_expressions = problem.first_order_system.rhs_expressions

    excluded = {independent, *state_symbols}
    parameter_symbols = _ordered_parameter_symbols(rhs_expressions, excluded)

    rhs_split_lines: list[str] = []
    for idx, state_symbol in enumerate(state_symbols):
        fn_name, state_order = _state_symbol_to_function_order(state_symbol)
        meaning = _function_derivative_text(fn_name, state_order, independent)
        rhs_split_lines.append(f"{str(state_symbol)} = y[{idx}]  # = {meaning}")

    parameter_block = ""
    if parameter_symbols:
        assignments = [
            f"    {str(symbol)} = params['{str(symbol)}']"
            for symbol in parameter_symbols
        ]
        parameter_block = "\n".join(assignments) + "\n"

    rhs_code_lines: list[str] = []
    for idx, expr in enumerate(rhs_expressions):
        fn_name, state_order = _state_symbol_to_function_order(state_symbols[idx])
        meaning = _function_derivative_text(fn_name, state_order + 1, independent)
        rhs_code_lines.append(f"{_expr_to_numpy_code(expr)},  # = {meaning}")
    rhs_code = "\n            ".join(rhs_code_lines)
    y0_code = ", ".join(str(float(sp.N(value))) for value in problem.y0)
    t0_code = str(float(sp.N(problem.t0)))

    script = f'''import numpy as np
from scipy.integrate import solve_ivp


def {function_name}(t_span, params=None, t_eval=None, **solve_kwargs):
    params = {{}} if params is None else dict(params)
{parameter_block}

    def fun(t, y):
        {"\n        ".join(rhs_split_lines)}
        return np.array([
            {rhs_code}
        ], dtype=float)

    y0 = np.array([{y0_code}], dtype=float)
    t0 = {t0_code}
    if abs(float(t_span[0]) - t0) > 1e-12:
        raise ValueError(f"t_span start {{t_span[0]}} does not match initial condition point {{t0}}")

    return solve_ivp(fun, t_span=t_span, y0=y0, t_eval=t_eval, **solve_kwargs)
'''
    return script


def generate_scipy_bvp_script(problem: BVPProblem, function_name: str = "solve_problem") -> str:
    """Generate a standalone SciPy BVP solver script as a Python string."""

    state_symbols = problem.first_order_system.state_symbols
    independent = problem.first_order_system.independent_variable
    unknown_symbols = problem.parameter_symbols
    rhs_expressions = problem.first_order_system.rhs_expressions
    bc_expressions = problem.boundary_residuals
    left_point = problem.left_boundary
    right_point = problem.right_boundary

    fun_excluded = {independent, *state_symbols, *unknown_symbols}
    bc_excluded = {
        *problem.left_state_symbols,
        *problem.right_state_symbols,
        *unknown_symbols,
    }
    fixed_params = sorted(
        set(_ordered_parameter_symbols(rhs_expressions, fun_excluded))
        | set(_ordered_parameter_symbols(bc_expressions, bc_excluded)),
        key=lambda symbol: str(symbol),
    )

    fun_state_lines: list[str] = []
    for idx, state_symbol in enumerate(state_symbols):
        fn_name, state_order = _state_symbol_to_function_order(state_symbol)
        meaning = _function_derivative_text(fn_name, state_order, independent)
        fun_state_lines.append(f"        {str(state_symbol)} = y[{idx}]  # = {meaning}")
    fun_unknown_lines = [
        f"        {str(symbol)} = p[{idx}]"
        for idx, symbol in enumerate(unknown_symbols)
    ]
    fun_fixed_lines = [
        f"    {str(symbol)} = params['{str(symbol)}']"
        for symbol in fixed_params
    ]

    bc_left_lines: list[str] = []
    for idx, symbol in enumerate(problem.left_state_symbols):
        base_name = str(symbol).removesuffix("_left")
        fn_name, order_token = base_name.rsplit("_", 1)
        meaning = _function_derivative_at_point_text(fn_name, int(order_token), left_point)
        bc_left_lines.append(f"        {str(symbol)} = ya[{idx}]  # = {meaning}")
    bc_right_lines: list[str] = []
    for idx, symbol in enumerate(problem.right_state_symbols):
        base_name = str(symbol).removesuffix("_right")
        fn_name, order_token = base_name.rsplit("_", 1)
        meaning = _function_derivative_at_point_text(fn_name, int(order_token), right_point)
        bc_right_lines.append(f"        {str(symbol)} = yb[{idx}]  # = {meaning}")
    bc_unknown_lines = [
        f"        {str(symbol)} = p[{idx}]"
        for idx, symbol in enumerate(unknown_symbols)
    ]

    fun_rhs_lines: list[str] = []
    for idx, expr in enumerate(rhs_expressions):
        fn_name, state_order = _state_symbol_to_function_order(state_symbols[idx])
        meaning = _function_derivative_text(fn_name, state_order + 1, independent)
        fun_rhs_lines.append(f"{_expr_to_numpy_code(expr)},  # = {meaning}")
    fun_rhs_code = "\n            ".join(fun_rhs_lines)

    state_symbol_to_human: dict[sp.Symbol, sp.Symbol] = {}
    for idx, state_symbol in enumerate(state_symbols):
        fn_name, order = _state_symbol_to_function_order(state_symbol)
        left_human = _function_derivative_at_point_text(fn_name, order, left_point)
        right_human = _function_derivative_at_point_text(fn_name, order, right_point)
        state_symbol_to_human[problem.left_state_symbols[idx]] = sp.Symbol(left_human)
        state_symbol_to_human[problem.right_state_symbols[idx]] = sp.Symbol(right_human)

    bc_code_lines: list[str] = []
    for expr in bc_expressions:
        human_expr = cast(sp.Expr, expr.xreplace(state_symbol_to_human))
        bc_code_lines.append(
            f"{_expr_to_numpy_code(expr)},  # = {sp.sstr(human_expr)}"
        )
    bc_code = "\n            ".join(bc_code_lines)

    guess_code = ",\n            ".join(_expr_to_numpy_code(expr) for expr in problem.initial_guess_expressions)
    p_guess_code = ", ".join(str(float(sp.N(value))) for value in problem.parameter_guess)

    script = f'''import numpy as np
from scipy.integrate import solve_bvp


def {function_name}(x_mesh, params=None, **solve_kwargs):
    params = {{}} if params is None else dict(params)
{"\n".join(fun_fixed_lines)}
    x_mesh = np.asarray(x_mesh, dtype=float)

    def fun(x, y, p):
{"\n".join(fun_unknown_lines + fun_state_lines)}
        return np.array([
            {fun_rhs_code}
        ], dtype=float)

    def bc(ya, yb, p):
{"\n".join(bc_unknown_lines + bc_left_lines + bc_right_lines)}
        return np.array([
            {bc_code}
        ], dtype=float)

    y_guess = np.array([
            {guess_code}
    ], dtype=float)
    p_guess = np.array([{p_guess_code}], dtype=float)

    return solve_bvp(fun, bc, x_mesh, y_guess, p=p_guess, **solve_kwargs)
'''
    return script
