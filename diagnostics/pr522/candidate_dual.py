from __future__ import annotations
def _solve_dual(self) -> FloatArray:
    r"""Solves the entropic-pooling dual via SciPy's Truncated Newton
    Constrained method. By exploiting the smooth Fenchel dual and its
    closed-form gradient, it operates in :math:`\mathbb{R}^k` (the number of
    constraints) rather than :math:`\mathbb{R}^T` (the number of scenarios),
    yielding an order-of-magnitude speedup over primal CVXPY interior-point
    solvers.

    Returns
    -------
    sample_weight : ndarray of shape (n_observations,)
       The updated posterior probability vector.

    References
    ----------
    .. [1] "Convex Optimization", Cambridge University Press, Section 5.2.3,
        Entropy maximization, Boyd & Vandenberghe (2004)

    .. [2] "I-Divergence Geometry of Probability Distributions and Minimization
        Problems", The Annals of Probability, Csiszar (1975)

    .. [3] "Fully Flexible Views: Theory and Practice",
            Risk, Meucci (2013).

    """
    n_observations, _ = self._returns.shape
    # Init constraints with sum(p)==1, rescaled by its norm
    # Has better convergence than the normalized form done inside the dual.
    a = [np.ones(n_observations).reshape(-1, 1) / np.sqrt(n_observations)]
    b = [np.array([1.0]) / np.sqrt(n_observations)]
    bounds = [(None, None)]
    for name, constraints in self._constraints.items():
        if constraints is not None:
            a.append(constraints[0])
            b.append(constraints[1])
            s = constraints[1].size
            match name:
                case "equality" | "cvar_equality":
                    bounds += [(None, None)] * s
                case "fixed_equality":
                    # Equivalent to relaxing the problem with slack variables with
                    # a norm penalty to avoid solver infeasibility that may arise
                    # from overly tight constraints from fixing the moments.
                    bounds += [(-1000, 1000)] * s
                case "inequality":
                    bounds += [(0, None)] * s
                # All constraint kinds created by this estimator are handled.
                case _:  # pragma: no cover
                    raise KeyError(f"constrain {name}")

    a = np.hstack(a)
    b = np.hstack(b)

    correction = None
    if self.correlation_views is not None and len(self.correlation_views):
        q = self._prior_sample_weight
        positive = q > 0
        log_q = np.full(q.shape, -np.inf)
        np.log(q, out=log_q, where=positive)
        correction = (q, positive, log_q, b - a.T @ q, np.log(np.finfo(float).max))

    def func(x: FloatArray) -> tuple[float, FloatArray]:
        """Computes the Fenchel dual of the entropic-pooling problem in its
        unnormalized form.

        Parameters
        ----------
        x : ndarray of shape (n_constraints,)
           Dual variables (Lagrange multipliers for each constraint).

        Returns
        -------
        obj : float
           Value of the dual objective.
        grad : ndarray of shape  (n_constraints,)
           Gradient of the dual objective.
        """
        if correction is not None:
            from scipy.special import logsumexp
            q, positive, log_q, prior_residual, limit = correction
            u = -a @ x - 1
            if logsumexp(log_q + u) > limit:
                raise SolverError("Dual problem with Solver 'TNC' failed: floating-point range")
            weighted_delta = np.zeros_like(q)
            ordinary = positive & (u <= limit)
            weighted_delta[ordinary] = q[ordinary] * np.expm1(u[ordinary])
            large = positive & ~ordinary
            weighted_delta[large] = np.exp(log_q[large] + u[large]) - q[large]
            obj = np.sum(weighted_delta[positive] - q[positive] * u[positive]) + prior_residual @ x
            grad = prior_residual - a.T @ weighted_delta
            return obj, grad
        z = self._prior_sample_weight * np.exp(-a @ x - 1)
        obj = z.sum() + x @ b
        grad = b - a.T @ z
        return obj, grad

    # We use TNC as it often outperforms L-BFGS-B on EP problems because
    # it builds (implicitly) true curvature information via Hessian-vector products,
    # whereas L-BFGS-B only ever uses a low-rank quasi-Newton approximation.
    # We set stepmx=1 to avoid large solver defaults when the bounds are not None.
    solver_params = (
        self.solver_params
        if self.solver_params is not None
        else {
            "maxfun": 5000,
            "ftol": 1e-11,
            "xtol": 1e-8,
            "gtol": 1e-8,
            "stepmx": 1,
        }
    )

    sol = sco.minimize(
        func,
        x0=np.zeros_like(b),
        jac=True,
        bounds=bounds,
        method="TNC",
        options=solver_params,
    )
    if not sol.success:
        raise SolverError(
            "Dual problem with Solver 'TNC' failed. This typically occurs when the "
            "specified views conflict or are overly extreme. Consider using a "
            "prior that generates more synthetic data for extreme views. You can "
            "also change `solver_params` or try another `solver` such as "
            f"'CLARABEL'. Solver error: {sol.message}"
        )
    sample_weight = self._prior_sample_weight * np.exp(-1 - a @ sol.x)
    # Handles numerical precision errors
    sample_weight = np.clip(sample_weight, 0, 1)
    sample_weight /= sample_weight.sum()
    return sample_weight
