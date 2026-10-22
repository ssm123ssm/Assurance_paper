import numpy as np
from manim import *
from scipy import stats

from model import EXAMPLE as R
from section5_cantelli import (BB, BH, C0, CEIL, GRID, N0, NH, NMAX, N_C, PMAX, RK, TB, TH, U, WTOP, XA, XL, Y0, YS,
                               apt, band, baseline, clipped, density, dpt, moments, npt, poly, ppt, prob_axis, shape,
                               vline, width_true, x_of)
from style import *

KS = 2 * np.log(1 / U)
S0 = 1 / (2 * np.sqrt(C0 + 1))
Z = np.sqrt(KS)
DSG = Z * S0
SG_CEIL = 2 * DSG
N_SG = int(np.ceil(KS / R.w_star ** 2 - (R.a + R.b) - 1))


def dashed(points, color, width=2.5):
    return DashedVMobject(poly(points, color, width), num_dashes=90, dashed_ratio=0.55)


class Section5SubGaussian(Narrated, Scene):

    def setup(self):
        self.init_slots()
        self.eq_center, self.note_center, self.slot_width = np.array([0, -3.4, 0]), np.array([0, 2.9, 0]), 12.0

    def construct(self):
        self.posteriors()
        self.proxy()
        self.chernoff()
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

        self.caption(r"Follow one posterior, with mean $\mu$")
        self.al, self.be = shape(Y0)
        self.mu, self.sd = moments(self.al, self.be)
        self.main = curves[Y0]
        self.peak = stats.beta.pdf(self.mu, self.al, self.be)
        self.mu_marks = VGroup(DashedLine(dpt(self.mu), dpt(self.mu, self.peak), color=MUTED, stroke_width=1.5,
                                          dash_length=0.06),
                               MathTex(r"\mu", font_size=26).next_to(dpt(self.mu, self.peak), UP, buff=0.1))
        self.play(*[FadeOut(curves[y]) for y in YS if y != Y0], FadeOut(labels), run_time=0.8)
        self.play(FadeIn(self.mu_marks), run_time=0.6)
        self.clear_slots()
        self.wait(1.0)

    def proxy(self):
        mu = self.mu
        self.caption(rf"Every beta posterior with concentration $c$ is sub-Gaussian")
        self.equation(r"\mathbb{E}\,e^{\lambda(X-\mu)}\le e^{\lambda^2 s^2/2}\ \text{for every }\lambda,"
                      r"\qquad s^2=\frac{1}{4(c+1)}")
        self.side_note(r"The proxy variance $s^2$ is from Marchal and Arbel (2017)")
        self.wait(3.0)

        self.caption(r"Its tails are bounded as if it were a normal with standard deviation $s$")
        normal = dashed([dpt(t, stats.norm.pdf(t, mu, S0)) for t in GRID], INK)
        h = 0.3 * self.peak
        s_bar = DoubleArrow(dpt(mu, h), dpt(mu + S0, h), buff=0, color=INK, stroke_width=2, tip_length=0.1,
                            max_tip_length_to_length_ratio=0.3)
        s_lab = MathTex("s", font_size=26).next_to(s_bar, UP, buff=0.06)
        n_lab = Tex(r"normal, sd $s$", font_size=22).next_to(dpt(mu + 1.6 * S0, stats.norm.pdf(1.6, 0, 1) / S0),
                                                             RIGHT, buff=0.15)
        self.play(Create(normal), FadeIn(n_lab), run_time=1.4)
        self.play(GrowFromCenter(s_bar), FadeIn(s_lab), run_time=0.8)
        self.equation(rf"s=\frac{{1}}{{2\sqrt{{c+1}}}}={S0:.3f}")
        self.side_note(r"$s$ is the standard deviation of the widest posterior, the one with mean 1/2")
        self.wait(3.0)
        self.proxy_parts = VGroup(normal, n_lab, s_bar, s_lab)

    def chernoff(self):
        al, be, mu = self.al, self.be, self.mu
        env = lambda th: np.exp(-(np.asarray(th) - mu) ** 2 / (2 * S0 ** 2))
        ptop = ValueTracker(1.0)

        self.caption(r"Markov's inequality turns each choice of $\lambda$ into a tail bound")
        self.bottom_axis = baseline(BB, ticks=True, name=r"\theta")
        title = Tex(r"probability beyond $\theta$\\away from $\mu$", font_size=20).rotate(PI / 2)
        title.next_to([XA - 0.75, BB + BH / 2, 0], LEFT, buff=0)
        yaxis = always_redraw(lambda: prob_axis(ptop.get_value()))
        self.clear_slots()
        self.play(FadeOut(self.top_axis[1:-1]), FadeOut(self.proxy_parts[1]), FadeIn(self.bottom_axis), FadeIn(yaxis),
                  FadeIn(title), run_time=1.0)
        self.equation(r"\Pr(X\ge\mu+d)\le e^{-\lambda d}\,\mathbb{E}\,e^{\lambda(X-\mu)}\le e^{-\lambda d+\lambda^2s^2/2}")
        lam = ValueTracker(0.0)
        right = GRID[GRID >= mu]

        def lam_curve():
            l = lam.get_value()
            return clipped(right, np.exp(-l * (right - mu) + l ** 2 * S0 ** 2 / 2), ptop.get_value(), MUTED, 2)

        def touched():
            hi = mu + lam.get_value() * S0 ** 2
            th = np.linspace(mu, max(hi, mu + 1e-4), 200)
            return clipped(th, env(th), ptop.get_value(), INK, 3)

        def touch_dot():
            d = lam.get_value() * S0 ** 2
            return Dot(ppt(mu + d, np.exp(-d ** 2 / (2 * S0 ** 2)), ptop.get_value()), radius=0.07, color=INK)

        curve, trace, dot = always_redraw(lam_curve), always_redraw(touched), always_redraw(touch_dot)
        lam_lab = Tex(r"one choice of $\lambda$", font_size=20, color=MUTED)
        lam_lab.add_updater(lambda m: m.next_to(dot.get_center(), UR, buff=0.12))
        self.add(curve, trace)
        self.play(FadeIn(dot), FadeIn(lam_lab), run_time=0.4)
        self.play(lam.animate.set_value(0.32 / S0 ** 2), run_time=6.0, rate_func=linear)
        self.wait(0.5)

        self.caption(r"The best $\lambda$ at each point traces a bell")
        self.play(lam.animate.set_value((1 - mu) / S0 ** 2), run_time=1.2, rate_func=linear)
        self.equation(r"\min_{\lambda}\,e^{-\lambda d+\lambda^2s^2/2}=e^{-d^2/(2s^2)}"
                      r"\quad\text{at}\quad\lambda=d/s^2")
        lam_lab.clear_updaters()
        self.play(FadeOut(curve), FadeOut(dot), FadeOut(lam_lab), run_time=0.6)
        left = GRID[GRID <= mu]
        left_env = poly([ppt(t, p, 1.0) for t, p in zip(left, env(left))], INK, 3)
        self.side_note(r"The same argument for $-X$ gives the left side")
        self.play(Create(left_env, rate_func=lambda t: smooth(t)), run_time=1.2)
        self.wait(1.0)
        trace.clear_updaters()
        dyn_env = always_redraw(lambda: clipped(GRID, env(GRID), ptop.get_value(), INK))
        self.remove(trace, left_env)
        self.add(dyn_env)
        env_lab = Tex("sub-Gaussian bound", font_size=24).next_to(ppt(mu + 0.1, env(mu + 0.1), 1.0), RIGHT, buff=0.12)
        self.play(FadeIn(env_lab), run_time=0.5)

        self.caption(r"The bell is not a density. Its height caps the tail beyond each point")
        self.clear_slots()
        self.wait(2.5)

        self.caption(r"The posterior's own tail probabilities stay under the bell")
        rt = always_redraw(lambda: clipped(right, stats.beta.sf(right, al, be), ptop.get_value(), PASS))
        lt = always_redraw(lambda: clipped(left, stats.beta.cdf(left, al, be), ptop.get_value(), PASS))
        true_lab = Tex("true tail", font_size=24, color=PASS).next_to(ppt(mu + 0.06, stats.beta.sf(mu + 0.06, al, be),
                                                                          1.0), LEFT, buff=0.15)
        self.play(FadeIn(rt), FadeIn(lt), FadeIn(true_lab), run_time=1.2)
        self.side_note(r"The two sides meet at different heights because the mean is not the median")
        self.wait(3.0)

        self.caption(r"Zoom in on the bottom fifth of the probability scale")
        self.play(ptop.animate.set_value(0.2), FadeOut(env_lab), FadeOut(true_lab), FadeOut(self.note), run_time=2.5)
        self.note = None
        five = VGroup(DashedLine(ppt(0, U, 0.2), ppt(1, U, 0.2), color=INK, stroke_width=1.5, dash_length=0.08),
                      MathTex(r"5\%", font_size=22).next_to(ppt(1, U, 0.2), RIGHT, buff=0.1))
        self.play(Create(five), run_time=0.8)
        self.wait(0.8)

        self.caption(rf"The bell falls to 5\% at $\mu\pm{Z:.2f}\,s$, which sets two fences")
        f_lo, f_hi = mu - DSG, mu + DSG
        self.fences = VGroup(vline(f_lo), vline(f_hi))
        self.fence_labels = VGroup(MathTex(rf"\mu-{Z:.2f}\,s", font_size=24).next_to(self.fences[0], UP, buff=0.08),
                                   MathTex(rf"\mu+{Z:.2f}\,s", font_size=24).next_to(self.fences[1], UP, buff=0.08))
        fdots = VGroup(Dot(ppt(f_lo, U, 0.2), radius=0.08, color=INK), Dot(ppt(f_hi, U, 0.2), radius=0.08, color=INK))
        self.play(FadeIn(fdots, scale=1.5), run_time=0.6)
        self.play(Create(self.fences), FadeIn(self.fence_labels), run_time=1.0)
        self.equation(rf"e^{{-d^2/(2s^2)}}=0.05\iff d=s\sqrt{{2\log 20}}={Z:.2f}\,s")
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

        self.caption(r"So the 90\% interval fits between the fences")
        self.equation(r"W(y,n)\le2\sqrt{2\log 20}\,s=\sqrt{\frac{2\log 20}{c+1}}")
        self.wait(3.0)
        for m in (dyn_env, rt, lt, yaxis):
            m.clear_updaters()
        self.chernoff_parts = VGroup(dyn_env, rt, lt, yaxis, title, five, fdots, qlines, qdots, gaps)

    def uniform(self):
        top = TB + TH + 0.1
        self.caption(r"The fences use $s$, so their width is the same for every count")
        self.play(FadeOut(self.chernoff_parts), FadeOut(self.bracket), FadeOut(self.fence_labels),
                  FadeOut(self.mu_marks), FadeOut(self.proxy_parts[0]), FadeOut(self.proxy_parts[2:]),
                  FadeOut(self.bottom_axis[-1]), run_time=0.8)
        self.clear_slots()
        self.remove(self.main, self.fences)
        axes = VGroup(Line([XA, BB, 0], [XA, BB + BH, 0], color=INK, stroke_width=2),
                      MathTex(r"\mu", font_size=28).next_to([XA + XL, BB, 0], RIGHT, buff=0.2),
                      Tex("interval width", font_size=20).rotate(PI / 2)
                      .next_to([XA - 0.75, BB + BH / 2, 0], LEFT, buff=0))
        for w in (0.3, 0.6):
            axes.add(Line(apt(0, w), apt(0, w) + 0.08 * LEFT, color=INK, stroke_width=2),
                     MathTex(f"{w:.1f}", font_size=22).next_to(apt(0, w), LEFT, buff=0.12))
        flat = Line(apt(0, SG_CEIL), apt(1, SG_CEIL), color=INK, stroke_width=3)
        flat_lab = MathTex(rf"\sqrt{{2\log 20/(c+1)}}={SG_CEIL:.3f}", font_size=24)
        flat_lab.next_to(apt(0, SG_CEIL), UP, buff=0.1).align_to([XA + 0.2, 0, 0], LEFT)
        mus = np.linspace(0.005, 0.995, 300)
        true = poly([apt(m, width_true(m * C0, (1 - m) * C0)) for m in mus], PASS, 3)
        counts = VGroup(*[Dot(apt((R.a + y) / C0, width_true(R.a + y, C0 - R.a - y)), radius=0.035, color=PASS)
                          for y in range(N0 + 1)])
        tlab = Tex("true width, one dot per count", font_size=22, color=PASS)
        tlab.next_to(apt(0.5, width_true(0.5 * C0, 0.5 * C0)), DOWN, buff=0.2)

        al = ValueTracker(self.al)

        def live():
            a_ = al.get_value()
            b_ = C0 - a_
            mu = a_ / C0
            lo, hi = stats.beta.ppf([U, 1 - U], a_, b_)
            return VGroup(density(a_, b_), vline(mu - DSG, lo=TB, hi=top), vline(mu + DSG, lo=TB, hi=top),
                          Line(dpt(lo) + 0.25 * DOWN, dpt(hi) + 0.25 * DOWN, color=PASS, stroke_width=6),
                          DashedLine(dpt(mu), apt(mu, SG_CEIL), color=MUTED, stroke_width=1.2, dash_length=0.05),
                          Dot(apt(mu, SG_CEIL), radius=0.08, color=INK),
                          Dot(apt(mu, hi - lo), radius=0.08, color=PASS))

        dyn = always_redraw(live)
        self.add(dyn)
        self.play(FadeIn(axes), Create(flat), FadeIn(flat_lab), Create(true), FadeIn(counts), FadeIn(tlab),
                  run_time=1.5)
        self.equation(r"W(y,n)\le\sqrt{\frac{2\log 20}{c+1}}\quad\text{for every }y")
        self.play(al.animate.set_value(R.a + 55), run_time=3.5)
        self.play(al.animate.set_value(R.a + 5), run_time=4.0)
        self.play(al.animate.set_value(C0 / 2), run_time=2.5)
        self.wait(1.0)

        self.caption(r"Cantelli's fences grow with $\sigma$ and reach 0.505 at mean 1/2")
        arch = dashed([apt(m, 2 * RK * np.sqrt(m * (1 - m) / (C0 + 1))) for m in mus], MUTED)
        arch_lab = Tex(r"Cantelli, $2\sqrt{19}\,\sigma$", font_size=22, color=MUTED).move_to(apt(0.5, 0.56))
        self.play(Create(arch), FadeIn(arch_lab), FadeOut(flat_lab), run_time=1.5)
        self.equation(rf"2\sqrt{{2\log 20}}\,s={2 * Z:.2f}\,s\qquad\text{{versus}}\qquad "
                      rf"2\sqrt{{19}}\,\sigma_{{\max}}={2 * RK:.2f}\,s")
        self.wait(3.5)
        dyn.clear_updaters()
        self.play(*[FadeOut(m) for m in (dyn, axes, flat, true, counts, tlab, arch, arch_lab,
                                         self.bottom_axis[:-1], self.top_axis[0], self.top_axis[-1])], run_time=0.8)
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
        ns = np.linspace(0, NMAX, 400)
        ceiling = poly([npt(n, np.sqrt(KS / (ab + n + 1))) for n in ns], INK, 3)
        lab = Tex("sub-Gaussian", font_size=22).next_to(npt(NMAX, np.sqrt(KS / (ab + NMAX + 1))), RIGHT, buff=0.12)
        self.play(FadeIn(axes), run_time=0.8)
        self.play(Create(ceiling), FadeIn(lab), run_time=2.5)
        self.equation(r"\text{ceiling}=\sqrt{\frac{2\log 20}{a+b+n+1}}")
        self.wait(1.0)

        self.caption(rf"From $n={N_SG}$ the ceiling is below $w^*$, so every count passes")
        wline = VGroup(DashedLine(npt(0, R.w_star), npt(NMAX, R.w_star), color=INK, stroke_width=2,
                                  dash_length=0.08),
                       MathTex(rf"w^*={R.w_star:.2f}", font_size=24).next_to(npt(NMAX, R.w_star), RIGHT, buff=0.1))
        self.play(Create(wline), run_time=0.8)
        zone = Rectangle(width=npt(NMAX, 0)[0] - npt(N_SG, 0)[0], height=NH, stroke_width=0)
        zone.set_fill(PASS, opacity=0.12).move_to(npt(N_SG, 0), aligned_edge=DL)
        drop = DashedLine(npt(N_SG, np.sqrt(KS / (ab + N_SG + 1))), npt(N_SG, 0), color=INK, stroke_width=2,
                          dash_length=0.06)
        zlab = VGroup(MathTex(rf"n={N_SG}", font_size=24).next_to(npt(N_SG, 0), UR, buff=0.1),
                      MathTex(r"A(n)=1", font_size=28, color=PASS).move_to(npt((N_SG + NMAX) / 2, 0.75)))
        self.play(FadeIn(zone), Create(drop), FadeIn(zlab), run_time=1.2)
        self.wait(2.0)

        self.caption(rf"Cantelli needs $n={N_C}$, and the true widths all pass from $n=18$")
        cant = dashed([npt(n, np.sqrt((1 - U) / U / (ab + n + 1))) for n in np.linspace((1 - U) / U - ab - 1, NMAX,
                                                                                         400)], MUTED)
        clab = Tex("Cantelli ceiling", font_size=22, color=MUTED).next_to(npt(120, np.sqrt((1 - U) / U / (ab + 121))),
                                                                          UP, buff=0.1)
        widest = [width_true(R.a + np.arange(n + 1), R.b + n - np.arange(n + 1)).max() for n in range(NMAX + 1)]
        true = poly([npt(n, w) for n, w in enumerate(widest)], PASS, 3)
        tlab = Tex("widest true interval", font_size=22, color=PASS).next_to(npt(120, widest[120]), DOWN, buff=0.12)
        self.play(Create(cant), FadeIn(clab), run_time=1.5)
        self.play(Create(true), FadeIn(tlab), run_time=1.5)
        self.wait(2.5)

        self.caption(r"The paper uses whichever constant is smaller")
        self.equation(r"K_u=\min\left\{\frac{1-u}{u},\,2\log\frac{1}{u}\right\}=\min\{19,\,5.99\},\qquad "
                      rf"n_{{\mathrm{{cert}}}}=\left\lceil\frac{{K_u}}{{(w^*)^2}}-(a+b)-1\right\rceil={N_SG}", font_size=27)
        self.side_note(r"Cantelli has the smaller constant only for credibility below about 43\%")
        self.wait(4.0)
