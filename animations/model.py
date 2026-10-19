import numpy as np
from scipy import stats

TOL = 1e-12


def width(a, b, tau):
    return stats.beta.ppf(1 - tau / 2, a, b) - stats.beta.ppf(tau / 2, a, b)


class Curve:

    def __init__(self, a, b, tau, w_star, q, n_max):
        self.a, self.b, self.tau, self.w_star, self.q, self.n_max = float(a), float(b), tau, w_star, q, n_max
        self.pmf, self.W, self.ok = [], [], []
        for n in range(n_max + 1):
            y = np.arange(n + 1)
            self.pmf.append(stats.betabinom.pmf(y, n, self.a, self.b))
            self.W.append(width(self.a + y, self.b + n - y, tau))
            self.ok.append(self.W[-1] <= w_star + TOL)
        self.A = np.array([m[ok].sum() for m, ok in zip(self.pmf, self.ok)])
        self.sus = np.minimum.accumulate(self.A[::-1])[::-1]

    def passing(self, n):
        return np.nonzero(self.ok[n])[0].tolist()

    def first(self):
        return int(np.nonzero(self.A >= self.q - TOL)[0][0])

    def first_sustained(self):
        return int(np.nonzero(self.sus >= self.q - TOL)[0][0])


def all_pass_from(curve, n_check):
    for n in range(curve.n_max, n_check + 1):
        y = np.arange(n + 1)
        if not np.all(width(curve.a + y, curve.b + n - y, curve.tau) <= curve.w_star + TOL):
            return False
    return True


EXAMPLE = Curve(2.5, 11.0, tau=0.10, w_star=0.29, q=0.90, n_max=20)

assert all_pass_from(EXAMPLE, 57)
