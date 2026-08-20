"""An alternative scoreline family: Weibull counts coupled by a Frank copula.

The production model is two Poisson counts glued together by the Dixon-Coles tau correction. That
specification makes two separate commitments, and the literature's proposed replacement (Boshnakov,
Kharrat & McHale 2017) changes both at once:

======================  ==========================  ==============================================
commitment              production                  replacement
======================  ==========================  ==============================================
marginal count law      Poisson (variance = mean)   Weibull count (a free shape parameter ``c``)
dependence device       tau, on four low cells      Frank copula, across the whole table
======================  ==========================  ==============================================

Both are represented here as independent choices, so the two can be measured apart rather than as
one bundle. ``marginal="poisson", dependence="tau"`` reproduces the production family exactly, and
:attr:`CountSpec.is_inert` says so.

The Weibull count distribution
------------------------------
If the waiting times between goals are Weibull with shape ``c`` rather than exponential, the number
of goals in a match is no longer Poisson. Its probabilities (McShane, Adrian, Bradlow & Fader 2008)
are

    P(N = n) = sum over j >= n of  (-1)**(j + n) * rate**j * alpha[j, n] / Gamma(c*j + 1)

with coefficients defined by the recursion

    alpha[j, 0]     = Gamma(c*j + 1) / Gamma(j + 1)
    alpha[j, n + 1] = sum over m in [n, j) of  alpha[m, n] * Gamma(c*(j - m) + 1) / Gamma(j - m + 1)

``c = 1`` gives back the Poisson exactly (every ``alpha[j, 0]`` is 1 and the series collapses to the
Poisson expansion), ``c < 1`` over-disperses and ``c > 1`` under-disperses. That is the whole point
of the extra parameter: football's goal process may be more or less *regular* than a Poisson, and
this is the parameter that would say so.

Everything is computed on the rescaled coefficients ``beta[j, n] = alpha[j, n] / Gamma(c*j + 1)``
rather than on ``alpha`` itself, because ``alpha`` overflows a float64 well before the series is
long enough to be accurate -- at ``c = 1.2`` and 60 terms it passes 1e23 -- while ``beta`` stays
close to ``1 / j!``. The recursion survives the rescaling as a plain matrix product,

    beta[:, n + 1] = K @ beta[:, n]

with a strictly lower-triangular ``K`` that depends only on ``c``. The pmf for a whole batch of
matches is then a single matrix product, and so is its derivative -- which is what makes fitting
this at a hundred parameters over a thousand barriers affordable at all.

The truncated grid is renormalised, unlike the production likelihood, and that asymmetry is
deliberate rather than an oversight. The tau correction is *exactly* mass-preserving over the full
support -- its four cell adjustments cancel to zero identically -- so Dixon-Coles needs no
normalising constant and the production code is right not to pay for one. A Weibull count truncated
at ``max_goals`` genuinely loses its tail, so this family computes the deficit and divides it out.
The deficit is reported, because a large one means the grid is too short for the fitted shape.

The Frank copula
----------------
    C(u, v) = -(1/k) * log(1 + (exp(-k*u) - 1)(exp(-k*v) - 1) / (exp(-k) - 1))

``k > 0`` is positive dependence, ``k < 0`` negative, and ``k -> 0`` is independence. Applied to
*discrete* margins it gives the joint mass of a cell as the copula measure of the rectangle its two
CDF steps span. That construction preserves the margins exactly whatever ``k`` does, which is
precisely the property tau lacks.

Frank is the choice rather than Gaussian or Gumbel because it is the only common single-parameter
copula that is symmetric and spans the *whole* dependence range including negative -- and the sign
of football's low-score dependence is an empirical question this project has already had to
measure twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import gammaln

POISSON = "poisson"
WEIBULL = "weibull"
TAU = "tau"
FRANK = "frank"

MARGINALS = (POISSON, WEIBULL)
DEPENDENCES = (TAU, FRANK)

# The shape at which the Weibull count IS the Poisson, and the dependence at which the Frank
# copula IS independence. Both are the inertness points of their respective axes.
UNIT_SHAPE = 1.0
INDEPENDENT_KAPPA = 0.0

# Below this the Frank expression divides by (exp(-k) - 1), which goes to zero, and loses all
# precision, so the independence limit is taken directly. 1e-6 is far tighter than any dependence
# football shows (a Kendall tau of 0.03 is k of about 0.27) and far looser than the point where
# float64 gives out.
_KAPPA_INDEPENDENCE_TOL = 1e-6

# Probabilities below this are treated as this when logged, so an invalid cell is loud but finite
# rather than an inf that destroys the optimiser's line search.
_PROB_FLOOR = 1e-300

# The series alternates in sign, so its accuracy is governed by cancellation: the answer is the
# small residue of terms far larger than itself. The ratio sum|term| / |sum term| IS that loss
# factor, and float64 carries ~16 digits, so a condition number of 1e10 leaves ~6 -- comfortably
# more than a likelihood needs and comfortably short of where the result becomes noise.
#
# Measured rather than assumed: at the shapes and rates football produces (rate <= 3.7 at the
# production model's widest forecast, shapes 0.85 to 1.3) the pmf matches an independent Poisson
# to 1e-15 and its derivative to 1e-8. At rate 12 the same code returns garbage with the wrong
# sign. A rate cap would have to be justified for every shape separately; the condition number is
# the quantity that actually decides, so it is what gets checked.
_MAX_CONDITION = 1e10

# The copula rectangle difference subtracts nearly-equal numbers in the far tail, where a cell's
# true mass is around 1e-17, so cells come back at -1e-15 of the table's peak. Frank is a genuine
# copula and its rectangle inequality holds exactly in real arithmetic, so anything at that scale
# is rounding and is clipped. A violation six orders of magnitude larger is not rounding, and
# raises. Measured: the worst observed noise across shapes and dependences is 2.6e-15 relative.
_COPULA_NEGATIVE_TOL = 1e-9

# How many distinct shapes to keep series kernels for. A gradient step evaluates the same shape
# several times over (once for the value, once for the analytic pass, twice more for each
# finite-differenced scalar), and a Poisson family never changes shape at all, so even a small
# cache turns the kernel build into a one-off.
_KERNEL_CACHE_SIZE = 8

# Sentinel for the one field that has no defensible default: the series length is a numerical
# accuracy choice, it belongs in config.yaml with the reasoning beside it, and a spec built without
# it should fail loudly rather than silently inherit a number nobody chose. The marginal and the
# dependence DO default, to the production pair, because that is a structural fact about which
# specification this family reduces to rather than a value anyone tuned.
_CONFIGURED = -1


class CountFamilyError(ValueError):
    """Raised when a scoreline family is asked for something it cannot represent."""


@dataclass(frozen=True)
class CountSpec:
    """Which marginal law and which dependence device a fit uses.

    ``n_series_terms`` truncates the Weibull count's alternating series. The terms behave like
    Poisson terms ``rate**j / j!``, so they die once ``j`` is comfortably past the rate; 60 is some
    40 terms beyond anything football produces and costs one 61x61 matrix product per evaluation.
    It is a configured value rather than a hard constant because a short series is the one way this
    family can be quietly wrong, and a run has to be able to widen it.
    """

    marginal: str = POISSON
    dependence: str = TAU
    n_series_terms: int = _CONFIGURED

    def __post_init__(self) -> None:
        if self.marginal not in MARGINALS:
            raise CountFamilyError(f"marginal must be one of {MARGINALS}; got {self.marginal!r}")
        if self.dependence not in DEPENDENCES:
            raise CountFamilyError(
                f"dependence must be one of {DEPENDENCES}; got {self.dependence!r}"
            )
        if self.n_series_terms == _CONFIGURED:
            raise CountFamilyError(
                "n_series_terms has no default: pass the value from model.seams.scoreline"
            )
        if self.n_series_terms < 1:
            raise CountFamilyError(f"n_series_terms must be positive; got {self.n_series_terms}")

    @property
    def is_inert(self) -> bool:
        """True when this family IS the production Poisson-and-tau specification."""
        return self.marginal == POISSON and self.dependence == TAU

    @property
    def fits_shape(self) -> bool:
        return self.marginal == WEIBULL

    @property
    def fits_kappa(self) -> bool:
        return self.dependence == FRANK

    @property
    def fits_rho(self) -> bool:
        return self.dependence == TAU

    def label(self) -> str:
        return f"{self.marginal}+{self.dependence}"


@lru_cache(maxsize=_KERNEL_CACHE_SIZE)
def series_kernel(shape: float, max_goals: int, n_terms: int) -> np.ndarray:
    """``(n_terms + 1, max_goals + 1)`` matrix ``W`` with ``P(N = n) = sum_j rate**j * W[j, n]``.

    Depends only on the shape, so it is reused across every match in a batch and, being cached,
    across every evaluation at the same shape -- which for a Poisson family is all of them.
    ``W[j, n]`` is ``(-1)**(j + n) * beta[j, n]`` in the notation of the module docstring.
    """
    if shape <= 0:
        raise CountFamilyError(f"Weibull shape must be positive; got {shape}")
    j = np.arange(n_terms + 1)
    lg_shaped = gammaln(shape * j + 1.0)
    lg_plain = gammaln(j + 1.0)

    beta = np.zeros((n_terms + 1, max_goals + 1))
    beta[:, 0] = np.exp(-lg_plain)

    rows, cols = np.meshgrid(j, j, indexing="ij")
    gap = np.clip(rows - cols, 0, None)  # only the strictly-lower triangle is ever read
    log_kernel = lg_shaped[cols] + lg_shaped[gap] - lg_shaped[rows] - lg_plain[gap]
    kernel = np.where(cols < rows, np.exp(log_kernel), 0.0)
    for n in range(max_goals):
        beta[:, n + 1] = kernel @ beta[:, n]

    # Built from the index vectors rather than by slicing the (n_terms+1)-square `rows`, which
    # silently assumes the series is at least as long as the goal grid. It need not be: a short
    # series is exactly the configuration a test has to be able to ask for in order to show that
    # the length matters.
    sign = np.where((j[:, None] + np.arange(max_goals + 1)[None, :]) % 2 == 0, 1.0, -1.0)
    kernel_out = sign * beta
    # Cached and therefore shared between callers: make it impossible to corrupt one fit's
    # coefficients from another.
    kernel_out.flags.writeable = False
    return kernel_out


def _monomials(rates: np.ndarray, n_terms: int) -> np.ndarray:
    """``(N, n_terms + 1)`` matrix of ``rate**j``, as ``exp(j * log(rate))``.

    A cumulative product was tried as a cheaper alternative and measured against this form on
    identical rates: the two agree to within their own last bit, so the choice is presentation
    rather than precision. What does move -- by two orders of magnitude between a rate of 1.2 and a
    rate of 3.5 -- is the *series'* accuracy, and that is the alternating sum's cancellation, which
    :func:`count_pmf` reports as a condition number rather than hides.
    """
    powers = np.arange(n_terms + 1, dtype=float)
    return np.exp(np.log(rates)[:, None] * powers[None, :])


def count_pmf(
    rates: np.ndarray, kernel: np.ndarray, *, want_gradient: bool = True
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """``(N, G+1)`` truncated pmf, its derivative with respect to ``log rate``, and a condition number.

    Neither array is renormalised here: the caller does that, because the normalising constant and
    its derivative are wanted separately -- the tail deficit is a reported diagnostic, not just a
    divisor.

    The third return value is ``sum|term| / |sum term|``, the cancellation loss of the alternating
    series and the honest way to know whether these numbers mean anything (see
    :data:`_MAX_CONDITION`). It is evaluated at the batch's **extreme rates only**: cancellation
    grows with the rate, because a larger rate pushes weight onto high-``j`` terms that are orders
    of magnitude above the answer they sum to, so the extremes bound the batch. Checking all N rows
    would cost a third of the routine's total work to refine a safety margin that already spans six
    orders of magnitude.
    """
    rates = np.asarray(rates, dtype=float)
    if np.any(rates <= 0):
        raise CountFamilyError("count rates must be positive")
    n_terms = kernel.shape[0] - 1
    monomials = _monomials(rates, n_terms)
    pmf = monomials @ kernel
    d_pmf = None
    if want_gradient:
        d_pmf = (monomials * np.arange(n_terms + 1)[None, :]) @ kernel

    extremes = _monomials(np.array([rates.min(), rates.max()]), n_terms)
    edge_pmf = extremes @ kernel
    magnitude = extremes @ np.abs(kernel)
    with np.errstate(divide="ignore", invalid="ignore"):
        condition = np.where(edge_pmf != 0.0, magnitude / np.abs(edge_pmf), np.inf)
    return pmf, d_pmf, float(np.nanmax(condition))


def normalise_pmf(
    pmf: np.ndarray, d_pmf: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Divide out the truncated grid's missing tail. Returns ``(pmf, d_pmf, total)``.

    ``total`` is handed back and reported: it is 1 to nine decimals for any sane shape, and a value
    far from 1 means the series or the goal grid is too short for the shape being explored.
    """
    total = pmf.sum(axis=1)
    normed = pmf / total[:, None]
    if d_pmf is None:
        return normed, None, total
    d_total = d_pmf.sum(axis=1)
    d_normed = (d_pmf - normed * d_total[:, None]) / total[:, None]
    return normed, d_normed, total


def frank_copula(u: np.ndarray, v: np.ndarray, kappa: float) -> np.ndarray:
    """``C(u, v)``, with the independence limit taken directly near ``kappa = 0``."""
    if abs(kappa) < _KAPPA_INDEPENDENCE_TOL:
        return u * v
    denom = np.expm1(-kappa)
    return -np.log1p(np.expm1(-kappa * u) * np.expm1(-kappa * v) / denom) / kappa


def frank_partial_u(u: np.ndarray, v: np.ndarray, kappa: float) -> np.ndarray:
    """``dC/du`` -- the conditional distribution of ``V`` at ``U = u``."""
    if abs(kappa) < _KAPPA_INDEPENDENCE_TOL:
        return np.broadcast_to(v, np.broadcast_shapes(np.shape(u), np.shape(v))).astype(float)
    denom = np.expm1(-kappa)
    s, t = np.expm1(-kappa * u), np.expm1(-kappa * v)
    return np.exp(-kappa * u) * t / (denom + s * t)


def _gather(grid: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``grid[i, index[i]]``, where a negative index means "one step below zero", which is 0."""
    rows = np.arange(len(index))
    safe = np.maximum(index, 0)
    return np.where(index < 0, 0.0, grid[rows, safe])


def joint_from_copula(
    pmf_home: np.ndarray,
    pmf_away: np.ndarray,
    kappa: float,
    x: np.ndarray,
    y: np.ndarray,
    d_home: np.ndarray | None = None,
    d_away: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Observed-cell probabilities under the Frank copula, plus their log-rate derivatives.

    For discrete margins the joint mass of a cell is the copula measure of the rectangle its two
    CDF steps span,

        P(x, y) = C(U[x], V[y]) - C(U[x-1], V[y]) - C(U[x], V[y-1]) + C(U[x-1], V[y-1])

    Only the cell that actually happened is ever formed, so this costs two cumulative sums rather
    than an (N, G+1, G+1) grid. Pass ``d_home`` and ``d_away`` (the normalised pmf derivatives) to
    get the derivatives back; omit them for a value-only call.
    """
    cdf_home = np.cumsum(pmf_home, axis=1)
    cdf_away = np.cumsum(pmf_away, axis=1)
    xi = np.asarray(x, dtype=int)
    yi = np.asarray(y, dtype=int)
    u1, u0 = _gather(cdf_home, xi), _gather(cdf_home, xi - 1)
    v1, v0 = _gather(cdf_away, yi), _gather(cdf_away, yi - 1)

    prob = (
        frank_copula(u1, v1, kappa)
        - frank_copula(u0, v1, kappa)
        - frank_copula(u1, v0, kappa)
        + frank_copula(u0, v0, kappa)
    )
    if d_home is None or d_away is None:
        zero = np.zeros_like(prob)
        return prob, zero, zero

    # dP/du at each corner of the rectangle, signed by which corner it is. The copula is
    # exchangeable, so the same routine serves the away side with its arguments swapped.
    p_u1 = frank_partial_u(u1, v1, kappa) - frank_partial_u(u1, v0, kappa)
    p_u0 = frank_partial_u(u0, v0, kappa) - frank_partial_u(u0, v1, kappa)
    p_v1 = frank_partial_u(v1, u1, kappa) - frank_partial_u(v1, u0, kappa)
    p_v0 = frank_partial_u(v0, u0, kappa) - frank_partial_u(v0, u1, kappa)

    d_cdf_home = np.cumsum(d_home, axis=1)
    d_cdf_away = np.cumsum(d_away, axis=1)
    d_lam = p_u1 * _gather(d_cdf_home, xi) + p_u0 * _gather(d_cdf_home, xi - 1)
    d_mu = p_v1 * _gather(d_cdf_away, yi) + p_v0 * _gather(d_cdf_away, yi - 1)
    return prob, d_lam, d_mu


def joint_from_tau(
    pmf_home: np.ndarray,
    pmf_away: np.ndarray,
    rho: float,
    lam: np.ndarray,
    mu: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    d_home: np.ndarray | None = None,
    d_away: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Dixon-Coles tau device applied to arbitrary marginals, with its normaliser.

    Against Poisson margins the four cell adjustments cancel exactly and the normaliser is 1; that
    identity is what lets the production likelihood skip it entirely. Against a Weibull margin they
    do not cancel, so the constant is computed and differentiated rather than assumed away -- which
    is the only reason this function exists separately from the production code path.
    """
    p0, p1 = pmf_home[:, 0], pmf_home[:, 1]
    q0, q1 = pmf_away[:, 0], pmf_away[:, 1]
    xi = np.asarray(x, dtype=int)
    yi = np.asarray(y, dtype=int)

    tau = np.ones_like(p0)
    tau = np.where((xi == 0) & (yi == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((xi == 0) & (yi == 1), 1.0 + lam * rho, tau)
    tau = np.where((xi == 1) & (yi == 0), 1.0 + mu * rho, tau)
    tau = np.where((xi == 1) & (yi == 1), 1.0 - rho, tau)

    correction = rho * (-lam * mu * p0 * q0 + lam * p0 * q1 + mu * p1 * q0 - p1 * q1)
    total = pmf_home.sum(axis=1) * pmf_away.sum(axis=1) + correction
    cell_home, cell_away = _gather(pmf_home, xi), _gather(pmf_away, yi)
    prob = tau * cell_home * cell_away / total
    if d_home is None or d_away is None:
        zero = np.zeros_like(prob)
        return prob, zero, zero

    dp0, dp1 = d_home[:, 0], d_home[:, 1]
    dq0, dq1 = d_away[:, 0], d_away[:, 1]
    # tau's own dependence on the rates, present only in the three cells that carry one. The
    # derivative is with respect to log lam, hence the extra factor of lam on every rate that
    # appears explicitly.
    d_tau_lam = np.where((xi == 0) & (yi == 0), -lam * mu * rho, 0.0)
    d_tau_lam = np.where((xi == 0) & (yi == 1), lam * rho, d_tau_lam)
    d_tau_mu = np.where((xi == 0) & (yi == 0), -lam * mu * rho, 0.0)
    d_tau_mu = np.where((xi == 1) & (yi == 0), mu * rho, d_tau_mu)

    d_corr_lam = rho * (
        -lam * mu * (dp0 * q0 + p0 * q0) + lam * (dp0 * q1 + p0 * q1)
        + mu * dp1 * q0 - dp1 * q1
    )
    d_corr_mu = rho * (
        -lam * mu * (p0 * dq0 + p0 * q0) + lam * p0 * dq1
        + mu * (p1 * dq0 + p1 * q0) - p1 * dq1
    )
    d_total_lam = d_home.sum(axis=1) * pmf_away.sum(axis=1) + d_corr_lam
    d_total_mu = pmf_home.sum(axis=1) * d_away.sum(axis=1) + d_corr_mu

    d_cell_home, d_cell_away = _gather(d_home, xi), _gather(d_away, yi)
    d_lam = (
        (d_tau_lam * cell_home * cell_away + tau * d_cell_home * cell_away) / total
        - prob * d_total_lam / total
    )
    d_mu = (
        (d_tau_mu * cell_home * cell_away + tau * cell_home * d_cell_away) / total
        - prob * d_total_mu / total
    )
    return prob, d_lam, d_mu


@dataclass(frozen=True)
class Marginals:
    """The two count distributions of a batch of matches, and how much to trust them.

    Held as its own object because the dependence parameters -- tau's rho and the copula's kappa --
    do not enter the marginals at all. Finite-differencing them therefore costs a rectangle
    difference rather than a rebuild of the entire pmf, which is the difference between a walk that
    takes a quarter of an hour and one that takes a day.
    """

    lam: np.ndarray
    mu: np.ndarray
    pmf_home: np.ndarray
    pmf_away: np.ndarray
    d_home: np.ndarray | None
    d_away: np.ndarray | None
    tail_deficit: float
    condition: float

    valid: bool = True

    @property
    def is_trustworthy(self) -> bool:
        """Whether these numbers carry enough precision to optimise a likelihood against."""
        return self.valid and self.condition <= _MAX_CONDITION


def marginals(
    spec: CountSpec,
    lam: np.ndarray,
    mu: np.ndarray,
    *,
    shape: float,
    max_goals: int,
    want_gradient: bool = True,
) -> Marginals:
    """Both sides' count distributions, renormalised over the truncated grid."""
    kernel = series_kernel(shape, max_goals, spec.n_series_terms)
    pmf_home, d_home, cond_home = count_pmf(lam, kernel, want_gradient=want_gradient)
    pmf_away, d_away, cond_away = count_pmf(mu, kernel, want_gradient=want_gradient)
    # A truncated series that has lost its precision returns negative probabilities and a total
    # near zero, and dividing by that total sends the CDF far outside [0, 1] -- which the Frank
    # copula then overflows on. The condition number catches the extreme cases; this catches the
    # rest, and it is the invariant that actually matters: these are probabilities.
    valid = bool(
        pmf_home.min() >= 0.0 and pmf_away.min() >= 0.0
        and pmf_home.sum(axis=1).min() > 0.0 and pmf_away.sum(axis=1).min() > 0.0
    )
    pmf_home, d_home, total_home = normalise_pmf(pmf_home, d_home)
    pmf_away, d_away, total_away = normalise_pmf(pmf_away, d_away)
    return Marginals(
        lam=lam,
        mu=mu,
        pmf_home=pmf_home,
        pmf_away=pmf_away,
        d_home=d_home,
        d_away=d_away,
        tail_deficit=float(
            max(np.abs(total_home - 1.0).max(), np.abs(total_away - 1.0).max())
        ),
        condition=max(cond_home, cond_away),
        valid=valid,
    )


def observed_cell_likelihood(
    spec: CountSpec,
    marg: Marginals,
    x: np.ndarray,
    y: np.ndarray,
    *,
    rho: float,
    kappa: float,
    want_gradient: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(prob, d_lam, d_mu)`` for the scoreline each match actually produced.

    The derivatives are of the *probability* with respect to the log rates, not of its log; the
    caller divides.
    """
    grad_home = marg.d_home if want_gradient else None
    grad_away = marg.d_away if want_gradient else None
    if want_gradient and (grad_home is None or grad_away is None):
        raise CountFamilyError(
            "a gradient was asked for from marginals that were built without one"
        )
    if spec.dependence == FRANK:
        return joint_from_copula(
            marg.pmf_home, marg.pmf_away, kappa, x, y, grad_home, grad_away
        )
    return joint_from_tau(
        marg.pmf_home, marg.pmf_away, rho, marg.lam, marg.mu, x, y, grad_home, grad_away
    )


def joint_grid(
    spec: CountSpec,
    lam: np.ndarray,
    mu: np.ndarray,
    *,
    shape: float,
    rho: float,
    kappa: float,
    max_goals: int,
) -> np.ndarray:
    """``(N, G+1, G+1)`` scoreline probabilities -- the prediction-time counterpart.

    Formed in full here because collapsing to home/draw/away needs every cell, unlike the
    likelihood, which only ever needs the one that happened.
    """
    marg = marginals(spec, lam, mu, shape=shape, max_goals=max_goals)
    if not marg.is_trustworthy:
        raise CountFamilyError(
            f"the count series cancelled to {marg.condition:.1e} of its own magnitude at shape "
            f"{shape}; these rates are outside the range it can represent"
        )
    pmf_home, pmf_away = marg.pmf_home, marg.pmf_away

    if spec.dependence == FRANK:
        zero = np.zeros((len(pmf_home), 1))
        cdf_home = np.cumsum(pmf_home, axis=1)
        cdf_away = np.cumsum(pmf_away, axis=1)
        upper_h = cdf_home[:, :, None]
        lower_h = np.concatenate([zero, cdf_home[:, :-1]], axis=1)[:, :, None]
        upper_a = cdf_away[:, None, :]
        lower_a = np.concatenate([zero, cdf_away[:, :-1]], axis=1)[:, None, :]
        joint = (
            frank_copula(upper_h, upper_a, kappa)
            - frank_copula(lower_h, upper_a, kappa)
            - frank_copula(upper_h, lower_a, kappa)
            + frank_copula(lower_h, lower_a, kappa)
        )
    else:
        joint = pmf_home[:, :, None] * pmf_away[:, None, :]
        joint[:, 0, 0] *= 1.0 - lam * mu * rho
        joint[:, 0, 1] *= 1.0 + lam * rho
        joint[:, 1, 0] *= 1.0 + mu * rho
        joint[:, 1, 1] *= 1.0 - rho

    floor = -_COPULA_NEGATIVE_TOL * float(joint.max())
    if joint.min() < floor:
        raise CountFamilyError(
            f"negative scoreline probability {joint.min():.3e} against a peak cell of "
            f"{joint.max():.3e}: the dependence parameter is outside the range this family "
            f"supports at these rates"
        )
    return np.clip(joint, 0.0, None) / np.clip(joint, 0.0, None).sum(axis=(1, 2), keepdims=True)


def log_probabilities(prob: np.ndarray) -> np.ndarray:
    """Log of the observed-cell probabilities, floored so an invalid cell is loud but finite."""
    return np.log(np.maximum(prob, _PROB_FLOOR))
