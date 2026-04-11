import scipy, re, sympy, numpy as np, sys, matplotlib.pyplot as plt
from ._common import *


class BVP(ODESolverBase):
    """
    A wrapper around scipy.integrate.solve_bvp() to solve a boundary value problem (BVP) for a system of ordinary differential equations (ODEs).
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_bvp.html for details.
    It simplifies the process of defining the ODEs, boundary conditions and initial guess.

    Regarding syntax
    ----------------
    Several of the parameters are passed as string containg python-like code representing mathematical expressions.
    These strings are not allowed to contain unicode characters. Otherwise, for the most part they are interpreted as normal python code and therefor have to obey python syntax rules, i.e.:
    - 'a' to the power of 'b' is 'a**b'
    - Mathematical brackets are '(' and ')'. '[]' and '{}' can be used but retain their 'python meaning'.
    - Names can contain digits but can't start with a digit.
    - Constants, variables, and function calls are supported, but they must be available in the current scope - typically as global definitions.

    

    The target variable is denoted as a valid python identifier like 'x', 'x1', 'z' etc.

    A target function is denoted as a valid python identifier like 'y', 'v', 'v1', 'v_1', 'foo', 'bar' etc.
    Invalid examples: 'y(x)' (the variable 'x' must not be specified with the function), '1y' (starts with a digit).

    The exception to the python syntax rules are the derivatives of the target function(s): There are two ways to denote the derivative:
    - Leibnitz notation: The derivative of 'y' by variable 'x' is denoted as 'dy/dx'. The second derivative is denoted either as 'd2y/dx2' or 'd^2y/dx^2' etc.
        Invalid examples: 'dy/dx(x)' (the variable 'x' must not be specified with the derivative), 'd2y/d3x' (incorrect derivative order)
    - Prime notation: The derivatives of 'y' are denoted as y', y'', y''' etc.
        Invalid example: "y'(x)" (the variable 'x' must not be specified with the derivative)
          
    Parameters
    ----------
    odes: The ordinary differential equation(s).
        Used to define a callback passed as parameter 'fun' to solve_bvp().
        See above for syntax rules.
        If only prime notation is used 'x' is used as the default variable name.
        E.g.: "y' = 2 * y + 1", "dy/dx = 2 * y + 1", "y + y' + d^2y/dx^2 = np.sin(x)"
            
    interval: The interval to solve the ode(s) in.
        Used to initialize a grid which is passed as parameter 'x' to solve_bvp().
        E.g.: (0, 1); ('a', 'b')

    bcs: The boundary condition(s) for the target function(s) and all but its highest order derivative.
        Used to define a callback passed as parameter 'bc' to solve_bvp().
        See above for general syntax rules.
        A bc must be an equation containing a target function and/or any of its derivatives (using prime notation) evaluated at one of the interval boundaries.
        Valid examples for interval (0, 1): "y(1)=5", "K_p * y'(0)**2 = 2 * p / (r * U**2) - y(0)"
        Invalid examples: "y=5" (no boundary specified), "dy/dx(1)=0" (can't use Leibnitz notation for bcs)

    initial_guess: The initial guess for the target function(s) and all but the highest order derivative(s) in alphabetical order (i.e. y, y', z, z', z'').
        Used to initialize the initial guess for the solution passed as parameter 'y' to solve_bvp().
        See above for general syntax rules.
        Can make use of the target variable (e.g. 'x'). The variable will have type 'numpy.ndarray'.
        Therefor, math functions from module 'numpy' should be prefered over those from module 'math'.
        If no initial guess is provided it is set to 0 for the target function(s) and all derivatives.
        Valid examples for target variable "x": (0, 17); "x"; ("x**2", "U - U * x")

    params: Names of the unknown parameters. If None (default), it is assumed that the problem doesn't depend on any parameters.
        E.g.: ("a", "b", "c")

    params_initial_guess: The initial guess for the unknown parameters (if any).
        Used to initialize the initial guess for the parameters passed as parameter 'p' to solve_bvp().
        If no initial guess is provided it is set to 0 for all parameters.
        E.g.: (0, 1, 2)
    """

    def __init__(self, odes : str | tuple[str,...],
                 interval : tuple[str | int | float, str | int | float],
                 bcs : str | tuple[str,...],
                 initial_guess : None | str | int | float | tuple[str | int | float,...] = None,
                 params : None | str | tuple[str,...] = None,
                 params_initial_guess : None | str | int | float | tuple[str | int | float,...] = None):
        
        super().__init__(odes, interval, default_variable="x")

        # initial guess
        if initial_guess is None:
            initial_guess = 0
        if not isinstance(initial_guess, (tuple, list)):
            initial_guess = [initial_guess] * self.n
        self.initial_guess = tuple(initial_guess)
        assert len(self.initial_guess) == self.n, f"wrong number of initial guesses ({len(self.initial_guess)}), should be {self.n}"
        
        # parameters
        if params is None:
            params = ()
        elif not isinstance(params, (tuple, list)):
            params = (params, )
        self.parameters = tuple(params)
        self.k = len(self.parameters)

        # initial guess for parameters
        if params_initial_guess is None:
            params_initial_guess = []
        elif not isinstance(params_initial_guess, (tuple, list)):
            params_initial_guess = [params_initial_guess]
        params_initial_guess = list(params_initial_guess)
        if len(params_initial_guess) < self.k:
            params_initial_guess.extend([0] * (self.k - len(params_initial_guess)))
        assert len(params_initial_guess) == self.k, f"wrong number of initial guesses for parameters ({len(params_initial_guess)}), should be {self.k}"
        self.params_initial_guess = tuple(params_initial_guess)

        # boundary conditions
        self.bcs : list[str] = []
        if isinstance(bcs, str):
            bcs = (bcs, )
        expected_bcs = self.n + self.k
        assert len(bcs) == expected_bcs, f"wrong number of boundary conditions ({len(bcs)}), should be {expected_bcs}"
        for bc in bcs:
            for target in self.targets:
                for match in re.findall(fr"\b{target.name}('*)\((\w+)\)", bc):
                    assert len(match) == 2, f"could not parse bc: '{bc}'"
                    apostrophes, parameter = match
                    derivative_order = len(apostrophes)
                    assert derivative_order < target.highest_order, f"bc contains derivative of order {derivative_order} (max is order {target.highest_order-1})"
                    if parameter == str(self.interval[0]):
                        postfix = "a"
                    elif parameter == str(self.interval[1]):
                        postfix = "b"
                    else:
                        raise ValueError(f"boundary condition parameter {parameter} is not an endpoint of interval {self.interval}")
                    bc = bc.replace(f"{target.name}{"'"*derivative_order}({parameter})", f"{self._derivative_python(target.name, derivative_order)}_{postfix}", 1)
            lhs, rhs = bc.split("=")
            self.bcs.append(f"{lhs.strip()} - ({rhs.strip()})")

        pass

    def generate_scipy_string(self, plot = False, steps = 50, **kwargs):
        """
        Generates a string containing python code which, if executed, solves the BVP.
        Either plug it into exec() or copy it to a new file and run it.
        If exec() is used the result of solve_bvp() can be accessed as 'solution'.

        Parameters
        ----------
        plot: If True, the solution is plotted afterwards (if the calculation finished successfully).
        steps: The number of steps to use for the solver.
        kwargs: Additional keyword arguments to pass to scipy.integrate.solve_bvp().
        """

        res = "import numpy as np, scipy\n\n"
        
        res += self._system_string(self.k > 0)

        if self.k > 0:
            res += "def bc(ya, yb, p):\n"
            res += f"{wh}{' '.join(f'{p},' for p in self.parameters)} = p\n"
        else:
            res += "def bc(ya, yb):\n"
        res += f"{wh}{"".join(f'{target}_a, ' for target in self._all_targets_python)}= ya\n"
        res += f"{wh}{"".join(f'{target}_b, ' for target in self._all_targets_python)}= yb\n"
        res += f"{wh}return [\n{wh * 2}"
        res += f",\n{wh * 2}".join(self.bcs)
        res += f"\n{wh}]\n\n"

        res += f"{self.variable} = np.linspace({self.interval[0]}, {self.interval[1]}, {steps})    # from, to, steps\n"
        res += f"initial_guess = np.zeros(({self.n}, {self.variable}.size))\n"
        for i, (guess, target) in enumerate(zip(self.initial_guess, self._all_targets_python)):
            if guess != 0:
                res += f"initial_guess[{i}] = {guess}    # initial guess for {target}\n"
        if self.k > 0:
            res += f"parameters_initial_guess = [{', '.join(f'{p}' for p in self.params_initial_guess)}]\n"
        res += "\n"

        res += f"solution = scipy.integrate.solve_bvp(system, bc, {self.variable}, initial_guess{", parameters_initial_guess" if self.k > 0 else ""}{''.join(f', {key}={val}' for key, val in kwargs.items())})\n"
        
        res += self._solution_string("x", self.k > 0)
        res += " ".join(derivative + f"," for derivative in self._all_derivatives) + " = solution.yp\n"
        # if self.k > 0:
        #     res += " ".join(f"{p}," for p in self.parameters) + f" = solution.p\n"

        res += self._error_and_plot_string(plot, True)

        return res
