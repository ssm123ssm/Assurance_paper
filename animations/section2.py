"""Section 2 of the paper. Passing set, precision assurance, and the first-crossing and sustained rules.

The example is the one in Figure 1, a Beta(2.5, 11) prior, 90% equal-tailed intervals, target width 0.29 and
required level q = 0.90. Render from this folder with ./render.sh, or directly,
    manim -qh --disable_caching section2.py Section2Assurance
"""

import numpy as np
from manim import *

from model import EXAMPLE as R, TOL
from stack_view import StackView, Tween, ViewScene
from style import *

N = R.n_max                 # rows n = 0..N
ROW = 11                    # the single study size of part 1, the first crossing
COL_X = -2.2                # count coordinate where each row's passing mass stacks

# Part 1 layout. The row is a flat bar chart with the width panel above it, on a shared count axis.
FLAT = dict(psi=0.0, elev=0.0, ux=0.5, ud=0.2, hscale=8.0, ox=-0.3, oy=-2.75)
W_LO, W_HI, W_Y0, W_Y1 = 0.15, 0.35, 0.35, 2.55
AXIS_X = -5.9
STRIP_X, STRIP_Y, STRIP_LEN, STRIP_H = 2.1, -1.25, 4.0, 0.36

# Part 2 camera, looking obliquely at the (count, n) floor.
OBLIQUE = dict(psi=32 * DEGREES, elev=26 * DEGREES, ux=0.3, ud=0.26, hscale=2.4, ox=0.0, oy=-1.1)


def w_screen(w):
    return W_Y0 + (w - W_LO) / (W_HI - W_LO) * (W_Y1 - W_Y0)


def curve_axes():
    ax = Axes(x_range=[0, N, 5], y_range=[0, 1, 0.2], x_length=10.0, y_length=4.6, tips=False,
              axis_config={"color": INK, "stroke_width": 2, "font_size": 22},
              y_axis_config={"numbers_to_include": [0, 0.5, 1.0],
                             "decimal_number_config": {"num_decimal_places": 1, "color": INK}},
              ).move_to([0.5, -0.65, 0])
    labels = VGroup(MathTex("A(n)", font_size=30).next_to(ax.y_axis, UP, buff=0.2),
                    MathTex("n", font_size=30).next_to(ax.x_axis, RIGHT, buff=0.2))
    numbers = VGroup(*[MathTex(str(t), font_size=22).next_to(ax.c2p(t, 0), DOWN, buff=0.2)
                       for t in range(0, N + 1, 5)])
    return ax, labels, numbers


def sustained_steps(ax, s):
    """Sustained assurance as steps, for the part of the n axis right of s. Green at or above q."""
    g = VGroup()
    for n in range(N, -1, -1):
        x0, x1 = max(n - 0.5, s, 0), min(n + 0.5, N)
        if x1 <= x0:
            break
        col = SUS if R.sus[n] >= R.q - TOL else MUTED
        g.add(Line(ax.c2p(x0, R.sus[n]), ax.c2p(x1, R.sus[n]), color=col, stroke_width=6))
        if n < N and x1 < N and R.sus[n + 1] != R.sus[n]:
            g.add(Line(ax.c2p(x1, R.sus[n]), ax.c2p(x1, R.sus[n + 1]), color=col, stroke_width=3))
    return g


class Section2Assurance(ViewScene):
    def construct(self):
        x, d, m, final = [], [], [], []
        for n in range(N + 1):
            for y in range(n + 1):
                x.append(y)
                d.append(n)
                m.append(R.pmf[n][y])
                final.append(PASS if R.ok[n][y] else FAIL)
        x, d, m = map(np.array, (x, d, m))
        start = [NEUTRAL if n == ROW else c for n, c in zip(d, final)]
        view = StackView(x, d, np.zeros(len(x)), start, x_lim=(-0.5, ROW + 0.5), d_lim=(-0.5, N + 0.5),
                         centre=(N / 2, N / 2), x_ticks=(0, 5, 10, 15, 20), d_ticks=(0, 5, 10, 15, 20),
                         floor_alpha=0.0, depth_alpha=0.0, num_alpha=0.0, **FLAT)
        self.views.append(view)
        self.add(view)

        self.one_size(view, x, d, m, final)
        ax, labels, numbers = curve_axes()
        self.all_sizes(view, x, d, m, ax)
        self.rules(ax, labels, numbers)

    # ------------------------------------------------------------------ part 1
    def one_size(self, view, x, d, m, final):
        row = d == ROW
        bar_x = {y: view.point(y, ROW)[0] for y in range(ROW + 1)}

        self.eq_center, self.note_center, self.slot_width = np.array([4.3, 0.45, 0]), np.array([4.3, 2.3, 0]), 4.3
        self.caption(rf"Plan a study of {ROW} observations")
        self.side_note(r"Prior $\mathrm{Beta}(2.5,\,11)$, $90\%$ equal-tailed intervals")
        counts = VGroup(*[MathTex(str(y), font_size=22).move_to(view.point(y, ROW) + 0.32 * DOWN)
                          for y in range(ROW + 1)])
        m_lab = MathTex(rf"m_{{{ROW}}}(y)", font_size=28).move_to([AXIS_X - 0.35, -1.55, 0])
        self.play(Tween(view, floor_alpha=1.0), FadeIn(counts), run_time=0.8)

        self.caption(r"Each possible count $y$ has a predictive probability")
        self.play(Tween(view, bm=np.where(row, m, 0.0), lag=0.6, lag_key=x / ROW), FadeIn(m_lab), run_time=2.0)
        self.wait(0.6)

        self.caption(r"Each count gives its own posterior interval width")
        w_axis = VGroup(Line([AXIS_X, W_Y0, 0], [AXIS_X, W_Y1, 0], color=INK, stroke_width=2))
        for t in (0.2, 0.3):
            yy = w_screen(t)
            w_axis.add(Line([AXIS_X - 0.08, yy, 0], [AXIS_X, yy, 0], color=INK, stroke_width=2),
                       MathTex(f"{t:.1f}", font_size=22).next_to([AXIS_X - 0.08, yy, 0], LEFT, buff=0.1))
        w_axis.add(MathTex(rf"W(y,{ROW})", font_size=28).move_to([AXIS_X + 0.55, W_Y1 + 0.35, 0]))
        widths = R.W[ROW]
        dots = VGroup(*[Dot([bar_x[y], w_screen(widths[y]), 0], radius=0.075, color=NEUTRAL)
                        for y in range(ROW + 1)])
        self.play(FadeIn(w_axis), run_time=0.6)
        self.play(LaggedStart(*[FadeIn(dot, shift=0.15 * DOWN) for dot in dots], lag_ratio=0.12), run_time=2.0)
        self.wait(0.5)

        self.caption(r"A count passes if its width is at most $w^*$, and fails otherwise")
        ys = w_screen(R.w_star)
        target = DashedLine([AXIS_X, ys, 0], [bar_x[ROW] + 0.5, ys, 0], color=INK, stroke_width=2.5,
                            dash_length=0.1)
        t_lab = MathTex(rf"w^*={R.w_star:.2f}", font_size=28).next_to(target, RIGHT, buff=0.15)
        self.play(Create(target), FadeIn(t_lab), run_time=1.0)
        cols = view.col.copy()
        for i in np.nonzero(row)[0]:
            cols[i] = ManimColor(final[i]).to_rgb()
        self.play(*[dot.animate.set_color(PASS if R.ok[ROW][y] else FAIL) for y, dot in enumerate(dots)],
                  Tween(view, col=cols), run_time=1.2)
        passing = R.passing(ROW)
        self.equation(rf"\mathcal S({ROW})=\{{{','.join(map(str, passing))}\}}")
        self.wait(1.0)

        self.caption(r"Assurance is the predictive probability of the passing set")
        frame = Rectangle(width=STRIP_LEN, height=STRIP_H, stroke_color=INK, stroke_width=1.5)
        frame.move_to([STRIP_X + STRIP_LEN / 2, STRIP_Y, 0])
        qx = STRIP_X + R.q * STRIP_LEN
        q_mark = VGroup(DashedLine([qx, STRIP_Y - 0.35, 0], [qx, STRIP_Y + 0.35, 0], color=INK, stroke_width=2,
                                   dash_length=0.06),
                        MathTex(rf"q={R.q:.2f}", font_size=24).move_to([qx, STRIP_Y - 0.6, 0]))
        s_lab = MathTex(rf"A({ROW})", font_size=28).next_to(frame, LEFT, buff=0.2)
        self.play(FadeIn(frame), FadeIn(q_mark), FadeIn(s_lab), run_time=0.6)
        left, segments = STRIP_X, VGroup()
        for y in passing:
            mass = R.pmf[ROW][y]
            h = mass * FLAT["hscale"]
            ghost = Rectangle(width=view.FX * FLAT["ux"], height=h, stroke_width=0).set_fill(PASS, opacity=0.9)
            ghost.move_to([bar_x[y], FLAT["oy"] + h / 2, 0])
            seg = Rectangle(width=mass * STRIP_LEN, height=STRIP_H, stroke_width=0).set_fill(PASS, opacity=0.9)
            seg.move_to([left + mass * STRIP_LEN / 2, STRIP_Y, 0])
            self.play(ReplacementTransform(ghost, seg), run_time=0.55)
            segments.add(seg)
            left += mass * STRIP_LEN
        self.equation(rf"A({ROW})=\sum_{{y\in\mathcal S({ROW})}}m_{{{ROW}}}(y)={R.A[ROW]:.3f}")
        self.side_note(rf"Above the required level $q={R.q:.2f}$")
        self.wait(2.0)

        self.part1 = VGroup(counts, m_lab, w_axis, dots, target, t_lab, frame, q_mark, s_lab, segments)

    # ------------------------------------------------------------------ part 2
    def all_sizes(self, view, x, d, m, ax):
        self.caption(r"Repeat at every sample size from 0 to 20")
        self.play(FadeOut(self.part1), run_time=0.8)
        self.clear_slots()
        self.play(Tween(view, x_lim=(-0.5, N + 0.5), depth_alpha=1.0, num_alpha=1.0, **OBLIQUE), run_time=2.5)
        self.play(Tween(view, bm=m, lag=0.7, lag_key=d / N), run_time=3.5)
        self.wait(1.5)

        self.caption(r"Stack the passing probability of each row into one column")
        tx, tz, top = x.astype(float), np.zeros(len(x)), np.ones(len(x))
        for n in range(N + 1):
            cum = 0.0
            for i in np.nonzero(d == n)[0]:
                if R.ok[n][x[i]]:
                    tx[i], tz[i] = COL_X, cum
                    cum += m[i]
                else:
                    top[i] = 0.0
        self.play(Tween(view, bx=tx, bz=tz, op=top, lag=0.6, lag_key=d / N), run_time=3.5)
        self.column_mask = top
        self.wait(1.5)

        self.caption(r"Seen from the side, the columns trace assurance against $n$")
        o = ax.c2p(view.centre[1], 0)
        unit_n = ax.c2p(1, 0)[0] - ax.c2p(0, 0)[0]
        unit_a = ax.c2p(0, 1)[1] - ax.c2p(0, 0)[1]
        self.play(Tween(view, psi=PI / 2, elev=0.0, ud=unit_n, hscale=unit_a, ox=o[0], oy=o[1],
                        count_alpha=0.0, depth_alpha=0.0), run_time=3.0)
        self.wait(1.0)

    # ------------------------------------------------------------------ part 3
    def rules(self, ax, labels, numbers):
        view = self.views[0]
        pts = [ax.c2p(n, R.A[n]) for n in range(N + 1)]
        falls = [n for n in range(N) if R.A[n + 1] < R.A[n] - TOL]
        segs = VGroup(*[Line(pts[n], pts[n + 1], color=INK, stroke_width=3) for n in range(N)])
        dots = VGroup(*[Dot(p, radius=0.06, color=INK) for p in pts])
        self.play(FadeIn(ax), FadeIn(labels), FadeIn(numbers), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(dot, scale=1.5) for dot in dots], lag_ratio=0.08), Create(segs),
                  run_time=2.5)
        self.wait(0.8)
        self.play(Tween(view, op=0.18 * self.column_mask), run_time=2.0)
        q_line = DashedLine(ax.c2p(0, R.q), ax.c2p(N, R.q), color=INK, stroke_width=2, dash_length=0.1)
        q_lab = MathTex(rf"q={R.q:.2f}", font_size=26).next_to(q_line, RIGHT, buff=0.12)
        self.play(Create(q_line), FadeIn(q_lab), run_time=0.8)

        self.eq_center, self.note_center, self.slot_width = np.array([1.0, 2.45, 0]), np.array([1.0, 1.85, 0]), 5.4
        self.caption(r"Assurance rises in steps and falls between them")
        self.play(*[segs[n].animate.set_color(FAIL).set_stroke(width=5) for n in falls],
                  *[dots[n + 1].animate.set_color(FAIL) for n in falls], Tween(view, op=0.0), run_time=1.5)
        self.wait(2.2)

        n_fc, n_sus = R.first(), R.first_sustained()
        self.caption(rf"The first-crossing rule selects $n={n_fc}$, where $A({n_fc})={R.A[n_fc]:.3f}$")
        tri = Triangle(color=INK, fill_opacity=1).scale(0.13).move_to(pts[n_fc] + 0.02 * UP)
        drop = DashedLine(pts[n_fc], ax.c2p(n_fc, 0), color=INK, stroke_width=1.8, dash_length=0.06)
        fc_lab = MathTex(rf"n^{{\mathrm{{fc}}}}={n_fc}", font_size=30).next_to(ax.c2p(n_fc, 0.55), LEFT, buff=0.15)
        self.play(FadeIn(tri, scale=1.5), Create(drop), FadeIn(fc_lab), run_time=1.0)
        self.wait(1.8)

        nxt = n_fc + 1
        self.caption(rf"One more observation gives $A({nxt})={R.A[nxt]:.3f}$, below $q$")
        a_lab = MathTex(rf"A({nxt})={R.A[nxt]:.3f}", font_size=28, color=FAIL).move_to(ax.c2p(17.2, 0.74))
        link = Line(pts[nxt] + 0.1 * DR, a_lab.get_corner(UL) + 0.05 * UL, color=FAIL, stroke_width=1.5)
        self.play(dots[nxt].animate.scale(1.8), Create(link), FadeIn(a_lab), run_time=0.8)
        self.wait(2.4)

        self.caption(r"Sustained assurance is the lowest assurance at $n$ or any larger $n$")
        self.equation(r"A^{\mathrm{sus}}(n)=\min_{m\ge n}A(m)")
        s = ValueTracker(N + 0.5)
        steps = always_redraw(lambda: sustained_steps(ax, s.get_value()))
        cursor = always_redraw(lambda: DashedLine(ax.c2p(max(s.get_value(), 0), 0), ax.c2p(max(s.get_value(), 0), 1),
                                                  color=MUTED, stroke_width=1.5, dash_length=0.06))
        self.add(steps, cursor)
        self.play(s.animate.set_value(-0.5), run_time=5.0, rate_func=linear)
        steps.clear_updaters()
        self.play(FadeOut(cursor), dots.animate.set_opacity(0.5), segs.animate.set_stroke(opacity=0.5),
                  run_time=0.5)
        self.wait(0.6)

        self.caption(rf"The sustained rule selects $n={n_sus}$, where assurance stays above $q$")
        sq = Square(side_length=0.2, color=SUS, fill_opacity=1).move_to(ax.c2p(n_sus, R.sus[n_sus]))
        drop2 = DashedLine(ax.c2p(n_sus, R.sus[n_sus]), ax.c2p(n_sus, 0), color=SUS, stroke_width=1.8,
                           dash_length=0.06)
        sus_lab = MathTex(rf"n^{{\mathrm{{sus}}}}={n_sus}", font_size=30, color=SUS).next_to(
            ax.c2p(n_sus, 0.55), RIGHT, buff=0.15)
        self.play(FadeIn(sq, scale=1.5), Create(drop2), FadeIn(sus_lab), run_time=1.0)
        self.wait(2.0)

        self.caption(rf"The first-crossing rule selects {n_fc} and the sustained rule selects {n_sus}")
        self.wait(4.0)
