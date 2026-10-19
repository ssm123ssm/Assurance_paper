"""Beta-binomial precision assurance for the Section 2 example of the paper.

Every number drawn on screen comes from here. One Beta(a, b) prior predicts the data and analyses them, and
intervals are equal-tailed at credibility 1 - tau, as in Section 2 of the paper.
"""

import numpy as np
from scipy import stats

TOL = 1e-12          # width comparison tolerance used in the paper


def width(a, b, tau):
    """Equal-tailed interval width of Beta(a, b) at credibility 1 - tau, broadcasting."""
    return stats.beta.ppf(1 - tau / 2, a, b) - stats.beta.ppf(tau / 2, a, b)


class Curve:
    """Predictive masses, widths, passing sets and assurance for n = 0..n_max."""

    def __init__(self, a, b, tau, w_star, q, n_max):
        self.a, self.b, self.tau, self.w_star, self.q, self.n_max = float(a), float(b), tau, w_star, q, n_max
        self.pmf, self.W, self.ok = [], [], []
        for n in range(n_max + 1):
            y = np.arange(n + 1)
            self.pmf.append(stats.betabinom.pmf(y, n, self.a, self.b))
            self.W.append(width(self.a + y, self.b + n - y, tau))
            self.ok.append(self.W[-1] <= w_star + TOL)
        self.A = np.array([m[ok].sum() for m, ok in zip(self.pmf, self.ok)])
        # The backward minimum is the sustained assurance once A = 1 beyond n_max.
        self.sus = np.minimum.accumulate(self.A[::-1])[::-1]

    def passing(self, n):
        return np.nonzero(self.ok[n])[0].tolist()

    def first(self):
        return int(np.nonzero(self.A >= self.q - TOL)[0][0])

    def first_sustained(self):
        return int(np.nonzero(self.sus >= self.q - TOL)[0][0])


def all_pass_from(curve, n_check):
    """True if every count passes at every n from curve.n_max to n_check. Beyond n_check the uniform bound of
    the paper (Proposition 1) takes over, so n_check should be at least its n_cert."""
    for n in range(curve.n_max, n_check + 1):
        y = np.arange(n + 1)
        if not np.all(width(curve.a + y, curve.b + n - y, curve.tau) <= curve.w_star + TOL):
            return False
    return True


# Figure 1 and Section 2 of the paper. Beta(2.5, 11) prior, 90% equal-tailed intervals, w* = 0.29, q = 0.90.
EXAMPLE = Curve(2.5, 11.0, tau=0.10, w_star=0.29, q=0.90, n_max=20)

# Proposition 1 gives n_cert = 57 here, and direct calculation covers 20 to 57, so the backward minimum over
# n <= 20 is the sustained assurance.
assert all_pass_from(EXAMPLE, 57)
