import numpy as np
from manim import *
from scipy import stats

from model import EXAMPLE as R
from style import *

U = R.tau / 2
K = (1 - U) / U
RK = np.sqrt(K)
N0 = 60
C0 = R.a + R.b + N0
Y0 = 15
YS = (5, 15, 30, 45, 55)
PMAX = 12.0
XA, XL = -5.4, 11.2
TB, TH = 0.55, 2.0
BB, BH = -2.55, 2.3
WTOP = 0.6
NX0, NL, NY0, NH, NMAX = -5.6, 11.0, -2.4, 4.4, 240
CEIL = np.sqrt(K / (C0 + 1))
N_C = int(np.ceil(K / R.w_star ** 2 - (R.a + R.b) - 1))
GRID = np.linspace(0.0005, 0.9995, 800)


def shape(y, n=N0):
    al = R.a + y
    return al, R.a + R.b + n - al


def moments(al, be):
    c = al + be
    mu = al / c
    return mu, np.sqrt(mu * (1 - mu) / (c + 1))


def x_of(theta):
    return XA + theta * XL


def dpt(theta, d=0.0):
    return np.array([x_of(theta), TB + min(d, PMAX) / PMAX * TH, 0])


def ppt(theta, p, ptop):
    return np.array([x_of(theta), BB + p / ptop * BH, 0])


def apt(mu, w):
    return np.array([x_of(mu), BB + w / WTOP * BH, 0])


def npt(n, w):
    return np.array([NX0 + n / NMAX * NL, NY0 + min(w, 1.0) * NH, 0])


def poly(points, color, width=2.5):
    return VMobject().set_points_as_corners(points).set_stroke(color, width=width).set_fill(opacity=0)


def clipped(thetas, ps, ptop, color, width=3):
    g, run = VGroup(), []
    for t, p in zip(thetas, ps):
        if p <= ptop:
            run.append(ppt(t, p, ptop))
            continue
        if len(run) > 1:
            g.add(poly(run, color, width))
        run = []
    if len(run) > 1:
        g.add(poly(run, color, width))
    return g


def density(al, be, color=INK, fill=0.12):
    pts = [dpt(t, d) for t, d in zip(GRID, stats.beta.pdf(GRID, al, be))]
    area = Polygon(dpt(GRID[0]), *pts, dpt(GRID[-1]), stroke_width=0).set_fill(color, opacity=fill)
    return VGroup(area, poly(pts, color))


def band(al, be, lo, hi):
    lo, hi = max(lo, GRID[0]), min(hi, GRID[-1])
    if hi - lo < 1e-4:
        return VMobject()
    th = np.linspace(lo, hi, 200)
    pts = [dpt(t, d) for t, d in zip(th, stats.beta.pdf(th, al, be))]
    return Polygon(dpt(lo), *pts, dpt(hi), stroke_width=0).set_fill(PASS, opacity=0.55)


def width_true(al, be):
    return stats.beta.ppf(1 - U, al, be) - stats.beta.ppf(U, al, be)


def baseline(y, ticks=False, name=None):
    g = VGroup(Line([XA, y, 0], [XA + XL, y, 0], color=INK, stroke_width=2))
    if ticks:
        for t in (0, 0.25, 0.5, 0.75, 1):
            g.add(Line([x_of(t), y, 0], [x_of(t), y - 0.08, 0], color=INK, stroke_width=2),
                  MathTex(f"{t:g}", font_size=22).next_to([x_of(t), y, 0], DOWN, buff=0.15))
    if name:
        g.add(MathTex(name, font_size=28).next_to([XA + XL, y, 0], RIGHT, buff=0.2))
    return g


def prob_axis(ptop):
    g = VGroup(Line([XA, BB, 0], [XA, BB + BH, 0], color=INK, stroke_width=2))
    for v in (ptop / 2, ptop):
        y = BB + v / ptop * BH
        g.add(Line([XA, y, 0], [XA - 0.08, y, 0], color=INK, stroke_width=2),
              MathTex(f"{round(v, 2):g}", font_size=22).next_to([XA - 0.08, y, 0], LEFT, buff=0.1))
    return g


def vline(theta, color=INK, dashed=True, lo=BB, hi=TB + TH + 0.1, width=2):
    if dashed:
        return DashedLine([x_of(theta), lo, 0], [x_of(theta), hi, 0], color=color, stroke_width=width,
                          dash_length=0.08)
    return Line([x_of(theta), lo, 0], [x_of(theta), hi, 0], color=color, stroke_width=width)


class Section5Cantelli(Narrated, Scene):

    def setup(self):
        self.init_slots()
        self.eq_center, self.note_center, self.slot_width = np.array([0, -3.4, 0]), np.array([0, 2.9, 0]), 12.0

    def construct(self):
        self.posteriors()
        self.cantelli()
        self.uniform()
        self.certificate()

    def posteriors(self):
        self.top_axis = baseline(TB, ticks=True, name=r"\theta")
        self.caption(rf"At $n={N0}$, each count $y$ gives a beta posterior")
        curves = {y: density(*shape(y)) for y in YS}
        labels = VGroup()
        for y in YS:
            al, be = shape(y)
            mu = moments(al, be)[0]
            labels.add(MathTex(rf"y={y}", font_size=22, color=MUTED)
                       .next_to(dpt(mu, stats.beta.pdf(mu, al, be)), UP, buff=0.1))
        self.play(Create(self.top_axis), run_time=0.8)
        self.play(LaggedStart(*[FadeIn(curves[y]) for y in YS], lag_ratio=0.25), FadeIn(labels), run_time=2.0)
        self.equation(rf"c=a+b+n={C0:g}\ \text{{for every }}y")
        self.wait(1.5)

        self.caption(r"$A(n)=1$ once every 90\% interval is narrower than $w^*$")
        self.equation(r"A(n)=1\iff W(y,n)\le w^*\ \text{for every }y")
        self.wait(2.0)

        self.caption(r"Follow one posterior, with mean $\mu$ and standard deviation $\sigma$")
        self.al, self.be = shape(Y0)
        self.mu, self.sd = moments(self.al, self.be)
        self.main = curves[Y0]
        peak = stats.beta.pdf(self.mu, self.al, self.be)
        h = 0.45 * peak
        sd_bar = DoubleArrow(dpt(self.mu, h), dpt(self.mu + self.sd, h), buff=0, color=INK, stroke_width=2,
                             tip_length=0.1, max_tip_length_to_length_ratio=0.3)
        self.mu_marks = VGroup(DashedLine(dpt(self.mu), dpt(self.mu, peak), color=MUTED, stroke_width=1.5,
                                          dash_length=0.06),
                               MathTex(r"\mu", font_size=26).next_to(dpt(self.mu, peak), UP, buff=0.1),
                               sd_bar, MathTex(r"\sigma", font_size=24).next_to(sd_bar, UP, buff=0.06))
        self.play(*[FadeOut(curves[y]) for y in YS if y != Y0], FadeOut(labels), run_time=0.8)
        self.play(FadeIn(self.mu_marks), run_time=0.8)
        self.equation(rf"\mu={self.mu:.3f},\qquad \sigma={self.sd:.3f}")
        self.wait(1.5)

    def cantelli(self):
        al, be, mu, sd = self.al, self.be, self.mu, self.sd
        env = lambda th: sd ** 2 / (sd ** 2 + (np.asarray(th) - mu) ** 2)
        right = lambda th: stats.beta.sf(th, al, be)
        left = lambda th: stats.beta.cdf(th, al, be)
        ptop = ValueTracker(1.0)

        self.caption(r"Cantelli bounds how much probability can lie far from the mean")
        self.bottom_axis = baseline(BB, ticks=True, name=r"\theta")
        title = Tex(r"probability beyond $\theta$\\away from $\mu$", font_size=20).rotate(PI / 2)
        title.next_to([XA - 0.75, BB + BH / 2, 0], LEFT, buff=0)
        yaxis = always_redraw(lambda: prob_axis(ptop.get_value()))
        self.play(FadeOut(self.top_axis[1:-1]), FadeIn(self.bottom_axis), FadeIn(yaxis), FadeIn(title),
                  run_time=1.0)
        static_env = poly([ppt(t, p, 1.0) for t, p in zip(GRID, env(GRID))], INK, 3)
        env_lab = Tex("Cantelli bound", font_size=24).next_to(ppt(mu + 0.09, env(mu + 0.09), 1.0), RIGHT, buff=0.12)
        self.play(Create(static_env), FadeIn(env_lab), run_time=1.6)
        self.equation(r"\Pr(X\ge\mu+d)\le\frac{\sigma^2}{\sigma^2+d^2},\qquad"
                      r"\Pr(X\le\mu-d)\le\frac{\sigma^2}{\sigma^2+d^2}")
        self.wait(2.0)

        self.caption(r"The bell is not a density. Its height caps the tail beyond each point")
        self.wait(2.5)

        self.caption(r"The posterior's own tail probabilities stay under the bell")
        dyn_env = always_redraw(lambda: clipped(GRID, env(GRID), ptop.get_value(), INK))
        self.remove(static_env)
        self.add(dyn_env)
        cr, cl = ValueTracker(mu + 1e-3), ValueTracker(mu - 1e-3)

        def trace(lo, hi, fn):
            th = np.linspace(lo, hi, 160)
            return clipped(th, fn(th), ptop.get_value(), PASS)

        rtrace = always_redraw(lambda: trace(mu, cr.get_value(), right))
        ltrace = always_redraw(lambda: trace(cl.get_value(), mu, left))
        rshade = always_redraw(lambda: band(al, be, cr.get_value(), 1.0))
        lshade = always_redraw(lambda: band(al, be, 0.0, cl.get_value()))

        def marker(tr, fn):
            t = tr.get_value()
            p = ptop.get_value()
            g = VGroup(vline(t, MUTED, width=1.2))
            g.add(Dot(ppt(t, min(fn(t), p), p), radius=0.07, color=PASS))
            g.add(Dot(ppt(t, min(env(t), p), p), radius=0.07, color=INK).set_opacity(float(env(t) <= p)))
            return g

        rmark = always_redraw(lambda: marker(cr, right))
        lmark = always_redraw(lambda: marker(cl, left))
        self.add(rshade, rtrace)
        self.bring_to_back(rshade)
        self.play(FadeIn(rmark), run_time=0.4)
        self.play(cr.animate.set_value(0.62), run_time=4.0, rate_func=linear)
        self.play(cr.animate.set_value(0.999), run_time=0.8, rate_func=linear)
        self.play(FadeOut(rmark), FadeOut(rshade), run_time=0.4)
        self.add(lshade, ltrace)
        self.bring_to_back(lshade)
        self.play(FadeIn(lmark), run_time=0.4)
        self.play(cl.animate.set_value(0.0), run_time=3.0, rate_func=linear)
        true_lab = Tex("true tail", font_size=24, color=PASS).next_to(ppt(mu + 0.06, right(mu + 0.06), 1.0), LEFT,
                                                                      buff=0.15)
        self.play(FadeOut(lmark), FadeOut(lshade), FadeIn(true_lab), run_time=0.6)
        self.side_note(r"The two sides meet at different heights because the mean is not the median")
        self.wait(3.0)

        self.caption(r"Zoom in on the bottom fifth of the probability scale")
        self.play(ptop.animate.set_value(0.2), FadeOut(env_lab), FadeOut(true_lab), FadeOut(self.note),
                  run_time=2.5)
        self.note = None
        five = VGroup(DashedLine(ppt(0, U, 0.2), ppt(1, U, 0.2), color=INK, stroke_width=1.5, dash_length=0.08),
                      MathTex(r"5\%", font_size=22).next_to(ppt(1, U, 0.2), RIGHT, buff=0.1))
        self.play(Create(five), run_time=0.8)
        self.wait(0.8)

        self.caption(r"The bell falls to 5\% at $\mu\pm\sqrt{19}\,\sigma$, which sets two fences")
        f_lo, f_hi = mu - RK * sd, mu + RK * sd
        self.fences = VGroup(vline(f_lo), vline(f_hi))
        self.fence_labels = VGroup(MathTex(r"\mu-\sqrt{19}\,\sigma", font_size=24).next_to(self.fences[0], UP, buff=0.08),
                                   MathTex(r"\mu+\sqrt{19}\,\sigma", font_size=24).next_to(self.fences[1], UP, buff=0.08))
        fdots = VGroup(Dot(ppt(f_lo, U, 0.2), radius=0.08, color=INK), Dot(ppt(f_hi, U, 0.2), radius=0.08, color=INK))
        self.play(FadeIn(fdots, scale=1.5), run_time=0.6)
        self.play(Create(self.fences), FadeIn(self.fence_labels), run_time=1.0)
        self.equation(r"\frac{\sigma^2}{\sigma^2+d^2}=0.05\iff d=\sqrt{19}\,\sigma")
        self.wait(2.0)

        self.caption(r"The true tails reach 5\% sooner, at the ends of the 90\% interval")
        q_lo, q_hi = stats.beta.ppf([U, 1 - U], al, be)
        qlines = VGroup(vline(q_lo, PASS, dashed=False, width=2.5), vline(q_hi, PASS, dashed=False, width=2.5))
        qdots = VGroup(Dot(ppt(q_lo, U, 0.2), radius=0.08, color=PASS), Dot(ppt(q_hi, U, 0.2), radius=0.08, color=PASS))
        gaps = VGroup(*[Arrow(ppt(q, U, 0.2) + 0.28 * UP, ppt(f, U, 0.2) + 0.28 * UP, buff=0, color=INK,
                              stroke_width=2.5, tip_length=0.14, max_tip_length_to_length_ratio=0.25)
                        for q, f in ((q_lo, f_lo), (q_hi, f_hi))])
        self.bracket = VGroup(Line(dpt(q_lo) + 0.25 * DOWN, dpt(q_hi) + 0.25 * DOWN, color=PASS, stroke_width=6),
                              Tex(r"90\% interval", font_size=22, color=PASS)
                              .next_to(dpt(q_hi) + 0.25 * DOWN, RIGHT, buff=0.15))
        self.play(FadeIn(qdots, scale=1.5), Create(qlines), run_time=1.0)
        self.play(GrowArrow(gaps[0]), GrowArrow(gaps[1]), FadeIn(self.bracket), run_time=0.9)
        self.wait(2.0)

        self.caption(r"So the 90\% interval always fits between the fences")
        self.equation(r"W(y,n)\le2\sqrt{19}\,\sigma")
        self.wait(2.5)
        dyn_env.clear_updaters()
        for m in (rtrace, ltrace, yaxis):
            m.clear_updaters()
        self.cantelli_parts = VGroup(dyn_env, rtrace, ltrace, yaxis, title, five, fdots, qlines, qdots, gaps)

    def uniform(self):
        mu0, sd0 = self.mu, self.sd
        self.caption(r"Each fence sits $\sqrt{19}\approx4.36$ standard deviations from the mean")
        self.play(FadeOut(self.cantelli_parts), FadeOut(self.bracket), FadeOut(self.bottom_axis[-1]), run_time=0.8)
        self.clear_slots()
        ry = TB - 0.3
        ruler = VGroup(Line([x_of(mu0), ry - 0.15, 0], [x_of(mu0), ry + 0.15, 0], color=INK, stroke_width=2.5))
        for side in (1, -1):
            edges = [mu0 + side * k * sd0 for k in range(5)] + [mu0 + side * RK * sd0]
            for i in range(5):
                ruler.add(Line([x_of(edges[i]), ry, 0], [x_of(edges[i + 1]), ry, 0],
                               color=INK if i % 2 == 0 else MUTED, stroke_width=7))
            for e in edges[1:]:
                ruler.add(Line([x_of(e), ry - 0.09, 0], [x_of(e), ry + 0.09, 0], color=INK, stroke_width=1.5))
        marks = VGroup(*[MathTex(t, font_size=20).move_to([x_of(mu0 + k * sd0), ry - 0.28, 0])
                         for k, t in ((1, r"\sigma"), (2, r"2\sigma"), (3, r"3\sigma"), (4, r"4\sigma"))])
        self.play(LaggedStart(*[Create(m) for m in ruler], lag_ratio=0.08), run_time=2.2)
        self.play(FadeIn(marks), run_time=0.6)
        self.equation(r"\text{fence width}=2\sqrt{19}\,\sigma")
        self.wait(2.0)

        self.caption(r"So the fence width depends on the posterior only through $\sigma$")
        self.wait(2.0)

        self.caption(rf"At $n={N0}$ every posterior has $c={C0:g}$, so $\sigma$ depends only on its mean")
        top = TB + TH + 0.1
        sig_top = WTOP / (2 * RK)
        axes = VGroup(Line([XA, BB, 0], [XA, BB + BH, 0], color=INK, stroke_width=2),
                      MathTex(r"\mu", font_size=28).next_to([XA + XL, BB, 0], RIGHT, buff=0.2))
        for v in (0.5, 1.0):
            axes.add(Line(apt(0, v * WTOP), apt(0, v * WTOP) + 0.08 * LEFT, color=INK, stroke_width=2))
        sig_ticks = VGroup(*[MathTex(f"{v * sig_top:.3f}", font_size=22).next_to(apt(0, v * WTOP), LEFT, buff=0.12)
                             for v in (0.5, 1.0)])
        wid_ticks = VGroup(*[MathTex(f"{v * WTOP:.1f}", font_size=22).next_to(apt(0, v * WTOP), LEFT, buff=0.12)
                             for v in (0.5, 1.0)])
        sig_title = Tex(r"standard deviation $\sigma$", font_size=20).rotate(PI / 2)
        sig_title.next_to([XA - 0.75, BB + BH / 2, 0], LEFT, buff=0)
        wid_title = Tex(r"fence width $2\sqrt{19}\,\sigma$", font_size=20).rotate(PI / 2)
        wid_title.next_to([XA - 0.75, BB + BH / 2, 0], LEFT, buff=0)
        sig_of = lambda m: np.sqrt(m * (1 - m) / (C0 + 1))
        on_axis = lambda m: apt(m, 2 * RK * sig_of(m))

        al = ValueTracker(self.al)
        seen = [mu0, mu0 + 1e-4]

        def live():
            a_ = al.get_value()
            b_ = C0 - a_
            mu, sd = moments(a_, b_)
            peak = stats.beta.pdf(mu, a_, b_)
            h = 0.45 * peak
            arrow = DoubleArrow(dpt(mu, h), dpt(mu + sd, h), buff=0, color=INK, stroke_width=2.5, tip_length=0.1,
                                max_tip_length_to_length_ratio=0.3)
            return VGroup(density(a_, b_), vline(mu - RK * sd, lo=TB, hi=top), vline(mu + RK * sd, lo=TB, hi=top),
                          arrow, MathTex(r"\sigma", font_size=24).next_to(arrow, UP, buff=0.06),
                          DashedLine(dpt(mu), on_axis(mu), color=MUTED, stroke_width=1.2, dash_length=0.05),
                          Dot(on_axis(mu), radius=0.08, color=INK))

        def trace():
            m = moments(al.get_value(), C0 - al.get_value())[0]
            seen[0], seen[1] = min(seen[0], m), max(seen[1], m)
            return poly([on_axis(x) for x in np.linspace(seen[0], seen[1], 200)], INK, 3)

        self.play(FadeOut(ruler), FadeOut(marks), FadeOut(self.fence_labels), FadeOut(self.mu_marks), run_time=0.6)
        self.remove(self.main, self.fences)
        dyn, tr = always_redraw(live), always_redraw(trace)
        self.add(dyn, tr)
        self.play(FadeIn(axes), FadeIn(sig_ticks), FadeIn(sig_title), run_time=0.8)
        self.equation(r"\sigma=\sqrt{\frac{\mu(1-\mu)}{c+1}}")
        self.play(al.animate.set_value(R.a + 55), run_time=3.5)
        self.play(al.animate.set_value(R.a + 5), run_time=4.0)
        full = poly([on_axis(x) for x in np.linspace(0.003, 0.997, 300)], INK, 3)
        tr.clear_updaters()
        self.play(Create(full), FadeOut(tr), run_time=1.0)
        self.wait(0.6)

        self.caption(r"$\sigma$ is largest when the mean is 1/2")
        self.play(al.animate.set_value(C0 / 2), run_time=2.5)
        cy = apt(0, CEIL)[1]
        ceil_line = DashedLine([XA, cy, 0], [XA + XL, cy, 0], color=INK, stroke_width=2, dash_length=0.08)
        sig_lab = MathTex(r"\sigma_{\max}=\frac{1}{2\sqrt{c+1}}=" + f"{1 / (2 * np.sqrt(C0 + 1)):.3f}",
                          font_size=24).next_to([XA, cy, 0], UP, buff=0.1).align_to([XA + 0.2, 0, 0], LEFT)
        self.play(Create(ceil_line), FadeIn(sig_lab), run_time=0.9)
        self.equation(r"\mu(1-\mu)\le\tfrac14\quad\Rightarrow\quad\sigma\le\frac{1}{2\sqrt{c+1}}")
        self.wait(2.5)

        self.caption(r"Multiplying by $2\sqrt{19}$ turns the $\sigma$ curve into the fence width")
        wid_lab = MathTex(r"\sqrt{19/(c+1)}=" + f"{CEIL:.3f}", font_size=24).next_to([XA, cy, 0], UP, buff=0.1)
        wid_lab.align_to([XA + 0.2, 0, 0], LEFT)
        self.play(Transform(sig_ticks, wid_ticks), Transform(sig_title, wid_title), Transform(sig_lab, wid_lab),
                  run_time=1.5)
        self.equation(r"2\sqrt{19}\,\sigma\le2\sqrt{19}\cdot\frac{1}{2\sqrt{c+1}}=\sqrt{\frac{19}{c+1}}")
        self.wait(3.0)

        self.caption(rf"Every 90\% interval at $n={N0}$ sits under this ceiling")
        true = poly([apt(m, width_true(m * C0, (1 - m) * C0)) for m in np.linspace(0.005, 0.995, 300)], PASS, 3)
        counts = VGroup(*[Dot(apt((R.a + y) / C0, width_true(R.a + y, C0 - R.a - y)), radius=0.035, color=PASS)
                          for y in range(N0 + 1)])
        tlab = Tex("true width, one dot per count", font_size=22, color=PASS)
        tlab.next_to(apt(0.5, width_true(0.5 * C0, 0.5 * C0)), DOWN, buff=0.2)
        self.play(Create(true), FadeIn(counts), FadeIn(tlab), run_time=1.5)
        self.equation(r"W(y,n)\le2\sqrt{19}\,\sigma\le\sqrt{\frac{19}{c+1}}\quad\text{for every }y")
        self.wait(3.5)
        dyn.clear_updaters()
        self.play(*[FadeOut(m) for m in (dyn, axes, sig_ticks, sig_title, full, ceil_line, sig_lab, true, counts,
                                         tlab, self.bottom_axis[:-1], self.top_axis[0], self.top_axis[-1])],
                  run_time=0.8)
        self.clear_slots()

    def certificate(self):
        self.caption(r"The ceiling falls as $n$ grows")
        axes = VGroup(Line(npt(0, 0), npt(NMAX, 0), color=INK, stroke_width=2),
                      Line(npt(0, 0), npt(0, 1.0), color=INK, stroke_width=2),
                      MathTex("n", font_size=28).next_to(npt(NMAX, 0), RIGHT, buff=0.15),
                      Tex("interval width", font_size=24).next_to(npt(0, 1.0), UP, buff=0.15)
                      .align_to(npt(0, 1.0), LEFT))
        for n in range(0, NMAX + 1, 50):
            axes.add(Line(npt(n, 0), npt(n, 0) + 0.08 * DOWN, color=INK, stroke_width=2),
                     MathTex(str(n), font_size=22).next_to(npt(n, 0), DOWN, buff=0.15))
        for w in (0.5, 1.0):
            axes.add(Line(npt(0, w), npt(0, w) + 0.08 * LEFT, color=INK, stroke_width=2),
                     MathTex(f"{w:.1f}", font_size=22).next_to(npt(0, w), LEFT, buff=0.15))
        ab = R.a + R.b
        ns = np.linspace(K - ab - 1, NMAX, 400)
        ceiling = poly([npt(n, np.sqrt(K / (ab + n + 1))) for n in ns], INK, 3)
        lab = MathTex(r"\sqrt{\frac{19}{a+b+n+1}}", font_size=26).next_to(npt(40, np.sqrt(K / (ab + 41))), UR,
                                                                            buff=0.1)
        self.play(FadeIn(axes), run_time=0.8)
        self.play(Create(ceiling), FadeIn(lab), run_time=2.5)
        mark = VGroup(Dot(npt(N0, CEIL), radius=0.07, color=INK),
                      MathTex(rf"n={N0}", font_size=22).next_to(npt(N0, CEIL), UR, buff=0.08))
        self.play(FadeIn(mark), run_time=0.6)
        self.wait(1.0)

        self.caption(rf"From $n={N_C}$ the ceiling is below $w^*$, so every count passes")
        wline = VGroup(DashedLine(npt(0, R.w_star), npt(NMAX, R.w_star), color=INK, stroke_width=2,
                                  dash_length=0.08),
                       MathTex(rf"w^*={R.w_star:.2f}", font_size=24).next_to(npt(NMAX, R.w_star), RIGHT, buff=0.1))
        self.play(Create(wline), run_time=0.8)
        zone = Rectangle(width=npt(NMAX, 0)[0] - npt(N_C, 0)[0], height=NH, stroke_width=0)
        zone.set_fill(PASS, opacity=0.12).move_to(npt(N_C, 0), aligned_edge=DL)
        drop = DashedLine(npt(N_C, np.sqrt(K / (ab + N_C + 1))), npt(N_C, 0), color=INK, stroke_width=2,
                          dash_length=0.06)
        zlab = VGroup(MathTex(rf"n={N_C}", font_size=24).next_to(npt(N_C, 0), DOWN, buff=0.45),
                      MathTex(r"A(n)=1", font_size=28, color=PASS).move_to(npt((N_C + NMAX) / 2, 0.75)))
        self.play(FadeIn(zone), Create(drop), FadeIn(zlab), run_time=1.2)
        self.wait(2.0)

        self.caption(r"The true widths all pass from $n=18$, so the bound is safe but loose")
        widest = [width_true(R.a + np.arange(n + 1), R.b + n - np.arange(n + 1)).max() for n in range(NMAX + 1)]
        true = poly([npt(n, w) for n, w in enumerate(widest)], PASS, 3)
        tlab = Tex("widest true interval", font_size=22, color=PASS).next_to(npt(70, widest[70]), UP, buff=0.12)
        self.play(Create(true), FadeIn(tlab), run_time=2.0)
        self.wait(2.5)

        self.caption(r"The certificate needs only $a$, $b$, the tail size and $w^*$")
        self.equation(r"n\ge\frac{1-u}{u\,(w^*)^2}-(a+b)-1,\qquad u=0.05")
        self.side_note(r"A sub-Gaussian bound gives a smaller certificate, $n\ge57$, for this example")
        self.wait(4.0)
