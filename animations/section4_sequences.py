from itertools import product
from math import comb

import numpy as np
from manim import *
from scipy.special import beta as beta_fn

from model import EXAMPLE as R
from style import *

N = 3
CH, CH1 = 0.32, 0.42
X0, W, G = -5.85, 11.0, 0.3
Y1, YT, YB = 1.1, 2.2, -1.55
EW, EGAP, IGAP, BGAP = 0.7, 0.35, 0.08, 0.7
EW4, PG, IG2, BG2, XC = 0.42, 0.06, 0.22, 0.5, 0.3
EW3 = 2 * EW4 + PG
TOPW = YB + 2 * CH
SOLID, TINT = 0.92, 0.2
LAB_X = -6.6
EQ1, NOTE1 = np.array([0, -1.75, 0]), np.array([0, -2.75, 0])
EQ2, NOTE2 = np.array([-3.3, 0.72, 0]), np.array([-3.3, -0.08, 0])
LEAK = (0,) * N + (1,)


def seq_prob(n, y):
    return beta_fn(R.a + y, R.b + n - y) / beta_fn(R.a, R.b)


def p_s(n, y):
    return (R.a + y) / (R.a + R.b + n)


def colour(n, y):
    return PASS if R.ok[n][y] else FAIL


def name(bits):
    return "".join("S" if b else "F" for b in bits)


def column(bits, w, left, top, color, ch=CH, alpha=1.0, upto=None):
    cells = VGroup()
    for r, b in enumerate(bits):
        a = alpha if upto is None or r < upto else 0.0
        c = Rectangle(width=max(w, 1e-3), height=ch, stroke_color=BG, stroke_width=1.5)
        c.set_fill(color, opacity=(SOLID if b else TINT) * a)
        cells.add(c.move_to([left + w / 2, top - (r + 0.5) * ch, 0]))
    return cells


def repaint(col, bits, color):
    return column(bits, col.width, col.get_left()[0], col.get_top()[1], color, col[0].height)


def edge_line(x, yc, h):
    return DashedLine([x, yc - h / 2 - 0.18, 0], [x, yc + h / 2 + 0.18, 0], color=INK, stroke_width=2,
                      dash_length=0.08)


def lay(widths, inner, outer, left):
    lefts, spans, x = [], [], left
    for ws in widths:
        a, row = x, []
        for i, w in enumerate(ws):
            row.append(x)
            x += w + (inner if i < len(ws) - 1 else 0)
        lefts.append(row)
        spans.append((a, x))
        x += outer
    return lefts, spans


def equal_layout(groups, w, inner, outer):
    total = sum(w * len(g) + inner * (len(g) - 1) for g in groups) + outer * (len(groups) - 1)
    return lay([[w] * len(g) for g in groups], inner, outer, XC - total / 2)


def row_labels(spans, yy, texts, font_size=24, min_gap=0.5):
    g, last = VGroup(), -10.0
    for (a, b), t in zip(spans, texts):
        x = (a + b) / 2
        if x - last < min_gap:
            continue
        g.add(MathTex(t, font_size=font_size).move_to([x, yy, 0]))
        last = x
    return g


def brace_under(a, b, yy, tex, font_size=26):
    br = Brace(Line([a, yy, 0], [b, yy, 0]), DOWN, buff=0.0, color=INK)
    return VGroup(br, MathTex(tex, font_size=font_size).next_to(br, DOWN, buff=0.1))


class Section4Sequences(Narrated, Scene):

    def setup(self):
        self.init_slots()

    def construct(self):
        self.enumerate()
        self.bundle()
        self.masses()
        self.passing()
        self.extend()
        self.rebundle()

    def enumerate(self):
        self.eq_center, self.note_center, self.slot_width = EQ1, NOTE1, 11.0
        self.seqs = list(product((0, 1), repeat=N))
        total = len(self.seqs) * EW + (len(self.seqs) - 1) * EGAP
        self.cols = {b: column(b, EW, -total / 2 + i * (EW + EGAP), Y1 + 1.5 * CH1, INK, CH1)
                     for i, b in enumerate(self.seqs)}
        self.caption(r"A study of 3 observations has 8 possible sequences")
        self.play(LaggedStart(*[FadeIn(c, shift=0.15 * DOWN) for c in self.cols.values()], lag_ratio=0.12),
                  run_time=2.0)
        self.side_note(r"Read each column from the top. Filled is a success (S), pale a failure (F).")
        self.wait(2.5)

    def bundle(self):
        self.caption(r"Group the sequences by their number of successes $y$")
        self.groups = [[b for b in self.seqs if sum(b) == y] for y in range(N + 1)]
        total = sum(EW * len(g) + IGAP * (len(g) - 1) for g in self.groups) + BGAP * N
        lefts, spans = lay([[EW] * len(g) for g in self.groups], IGAP, BGAP, -total / 2)
        top = Y1 + 1.5 * CH1
        self.play(*[self.cols[b].animate(path_arc=0.6).move_to([lefts[y][k] + EW / 2, Y1, 0])
                    for y, g in enumerate(self.groups) for k, b in enumerate(g)], run_time=2.0)
        self.ylabels = VGroup(*[MathTex(str(y), font_size=26).move_to([(a + b) / 2, top + 0.3, 0])
                                for y, (a, b) in enumerate(spans)])
        self.yhead = MathTex("y", font_size=26).move_to([spans[0][0] - 0.5, top + 0.3, 0])
        self.play(FadeIn(self.ylabels), FadeIn(self.yhead), run_time=0.6)
        self.wait(0.6)

        self.caption(r"Bundle $y$ holds $\binom{3}{y}$ sequences")
        yb = Y1 - 1.5 * CH1 - 0.45
        self.counts = VGroup(*[brace_under(a, b, yb, rf"\binom{{3}}{{{y}}}={comb(N, y)}")
                               for y, (a, b) in enumerate(spans)])
        self.play(LaggedStart(*[FadeIn(c) for c in self.counts], lag_ratio=0.25), run_time=1.4)
        self.wait(1.5)

    def masses(self):
        self.caption(r"Before any data, sequences with the same count are equally likely")
        probs = VGroup(*[MathTex(f"{seq_prob(N, sum(b)):.4f}", font_size=20).next_to(self.cols[b], DOWN, buff=0.12)
                         for b in self.seqs])
        self.clear_slots()
        self.play(LaggedStart(*[FadeIn(t) for t in probs], lag_ratio=0.06), run_time=1.2)
        self.equation(r"\Pr(\mathrm{SFF})=\Pr(\mathrm{FSF})=\Pr(\mathrm{FFS})"
                      r"=\frac{B(a+1,\,b+2)}{B(a,b)}=0.1088")
        self.wait(2.5)

        self.caption(r"The mass of bundle $y$ is $\binom{3}{y}$ times one sequence's probability")
        products = VGroup()
        for y, c in enumerate(self.counts):
            two = VGroup(MathTex(rf"{comb(N, y)}\times{seq_prob(N, y):.4f}", font_size=24),
                         MathTex(rf"={R.pmf[N][y]:.3f}", font_size=24)).arrange(DOWN, buff=0.1)
            products.add(two.next_to(c[0], DOWN, buff=0.1))
        self.play(*[ReplacementTransform(c[1], p) for c, p in zip(self.counts, products)], run_time=1.2)
        self.equation(r"m_3(y)=\underbrace{\binom{3}{y}}_{\text{sequences}}\;"
                      r"\underbrace{\frac{B(a+y,\,b+3-y)}{B(a,b)}}_{\text{each sequence}}", font_size=30)
        self.side_note(r"This is the beta-binomial predictive probability of count $y$")
        self.wait(2.5)

        self.caption(r"Draw each sequence with width equal to its probability")
        top = Y1 + 1.5 * CH
        lefts, spans = lay([[seq_prob(N, y) * W] * len(g) for y, g in enumerate(self.groups)], 0.0, G, X0)
        self.geom, anims = {}, []
        for y, g in enumerate(self.groups):
            w = seq_prob(N, y) * W
            for k, b in enumerate(g):
                self.geom[b] = (lefts[y][k], w, y)
                anims.append(Transform(self.cols[b], column(b, w, lefts[y][k], top, INK)))
        self.lvl = MathTex(rf"n={N}", font_size=28).move_to([LAB_X, Y1, 0])
        self.play(FadeOut(probs), FadeOut(VGroup(*[c[0] for c in self.counts])), FadeOut(products), run_time=0.6)
        self.clear_slots()
        self.play(*anims, *[t.animate.move_to([(a + b) / 2, top + 0.3, 0]) for t, (a, b) in zip(self.ylabels, spans)],
                  self.yhead.animate.move_to([LAB_X, top + 0.3, 0]), FadeIn(self.lvl), run_time=2.2)
        self.spans3 = spans
        self.mass_row = VGroup(MathTex(r"m_3(y)", font_size=24).move_to([LAB_X, Y1 - 1.5 * CH - 0.28, 0]),
                               row_labels(spans, Y1 - 1.5 * CH - 0.28, [f"{m:.3f}" for m in R.pmf[N]]))
        self.play(FadeIn(self.mass_row), run_time=0.6)
        self.caption(r"The bundle widths are the masses $m_3(y)$, and they add to 1")
        self.wait(2.5)

    def passing(self):
        self.caption(r"Check every count against the width target at $n=3$ and $n=4$")
        scale, dy, row0 = 7.0, 0.36, -0.7
        panels, bars = VGroup(), []
        for n, cx in ((N, -3.4), (N + 1, 3.4)):
            x0 = cx - 2.1
            bx = x0 + 0.55
            tx = bx + R.w_star * scale
            bottom = row0 - dy * n - 0.22
            panels.add(MathTex("y", font_size=24).move_to([x0 + 0.25, row0 + 0.42, 0]),
                       MathTex(rf"W(y,{n})", font_size=24).move_to([bx, row0 + 0.42, 0], aligned_edge=LEFT),
                       DashedLine([tx, bottom, 0], [tx, row0 + 0.25, 0], color=INK, stroke_width=1.5,
                                  dash_length=0.06),
                       MathTex(rf"w^*={R.w_star:.2f}", font_size=22).next_to([tx, bottom, 0], DOWN, buff=0.08))
            for y in range(n + 1):
                ry, w = row0 - dy * y, R.W[n][y]
                bar = Rectangle(width=w * scale, height=0.2, stroke_width=0)
                bar.set_fill(colour(n, y), opacity=0.9).move_to([bx, ry, 0], aligned_edge=LEFT)
                bars.append(bar)
                panels.add(MathTex(str(y), font_size=22).move_to([x0 + 0.25, ry, 0]),
                           MathTex(f"{w:.3f}", font_size=20).move_to([bx + 0.4 * scale + 0.25, ry, 0]))
        self.play(FadeIn(panels), run_time=0.6)
        self.play(LaggedStart(*[GrowFromEdge(b, LEFT) for b in bars], lag_ratio=0.1), run_time=2.0)
        self.wait(1.5)

        self.caption(r"At both sizes only count 0 passes")
        self.eq_center = np.array([0, -3.35, 0])
        self.equation(r"\mathcal S(3)=\mathcal S(4)=\{0\}")
        self.edge3 = edge_line(X0 + R.pmf[N][0] * W + G / 2, Y1, N * CH)
        self.play(*[Transform(self.cols[b], repaint(self.cols[b], b, colour(N, sum(b)))) for b in self.seqs],
                  Create(self.edge3), run_time=1.0)
        self.wait(2.0)

        self.caption(r"Assurance at $n=3$ is the mass of bundle 0")
        self.play(FadeOut(panels), FadeOut(VGroup(*bars)), run_time=0.6)
        self.eq_center = EQ1
        self.equation(rf"A(3)=m_3(0)=\Pr(\mathrm{{FFF}})={R.A[N]:.3f}")
        self.wait(2.0)

    def extend(self):
        mid = TOPW - 2 * CH
        self.caption(r"Redraw the sequences at equal width, keeping their colors")
        self.clear_slots()
        upper = VGroup(*self.cols.values(), self.ylabels, self.yhead, self.lvl, self.mass_row, self.edge3)
        self.play(upper.animate.shift((YT - Y1) * UP), run_time=1.2)
        lefts, spans = equal_layout(self.groups, EW3, IG2, BG2)
        self.work, self.wleft, anims = {}, {}, []
        for y, g in enumerate(self.groups):
            for k, b in enumerate(g):
                c = self.cols[b].copy()
                self.add(c)
                self.work[b], self.wleft[b] = c, lefts[y][k]
                anims.append(Transform(c, column(b, EW3, lefts[y][k], TOPW, colour(N, y))))
        self.wlvl = MathTex(rf"n={N}", font_size=28).move_to([LAB_X, mid, 0])
        self.wy = VGroup(*[MathTex(str(y), font_size=26).move_to([(a + b) / 2, TOPW + 0.3, 0])
                           for y, (a, b) in enumerate(spans)])
        self.wedge = edge_line(spans[0][1] + BG2 / 2, mid, (N + 1) * CH)
        self.play(*anims, run_time=2.0)
        self.play(FadeIn(self.wlvl), FadeIn(self.wy), Create(self.wedge), run_time=0.8)
        self.wait(1.5)

        self.caption(r"A fourth observation splits each sequence into two, giving 16")
        self.kids, half, full = {}, [], []
        for b in self.seqs:
            for last in (0, 1):
                m = b + (last,)
                x = self.wleft[b] + last * (EW4 + PG)
                kid = column(m, EW4, x, TOPW, colour(N, sum(b)), alpha=0.0)
                self.add(kid)
                half.append(Transform(kid, column(m, EW4, x, TOPW, colour(N, sum(b)), upto=N)))
                full.append(Transform(kid, column(m, EW4, x, TOPW, colour(N, sum(b)))))
                self.kids[m] = kid
        self.play(*[FadeOut(c) for c in self.work.values()], *half, FadeOut(self.wy), run_time=1.0)
        self.play(*full, Transform(self.wlvl, MathTex(rf"n={N + 1}", font_size=28).move_to(self.wlvl)),
                  run_time=0.9)
        self.wait(1.0)

        self.caption(r"The two children split the parent's probability by $1-p_3(y)$ and $p_3(y)$")
        fy = TOPW - (N + 1) * CH - 0.22
        self.factors = VGroup(*[MathTex(f"{p_s(N, sum(m[:N])) if m[N] else 1 - p_s(N, sum(m[:N])):.2f}",
                                        font_size=18).move_to([k.get_x(), fy, 0]) for m, k in self.kids.items()])
        self.eq_center, self.note_center, self.slot_width = EQ2, NOTE2, 5.6
        self.play(LaggedStart(*[FadeIn(t) for t in self.factors], lag_ratio=0.04), run_time=1.2)
        self.equation(r"p_3(y)=\frac{a+y}{a+b+3}")
        self.side_note(r"$\Pr(s\mathrm{F})+\Pr(s\mathrm{S})=\Pr(s)$")
        self.wait(3.0)

        self.caption(r"Ordered by parent, the 16 new counts are mixed")
        self.klabels = {m: MathTex(str(sum(m)), font_size=22).move_to([k.get_x(), TOPW + 0.3, 0])
                        for m, k in self.kids.items()}
        self.play(LaggedStart(*[FadeIn(t) for t in self.klabels.values()], lag_ratio=0.05), run_time=1.2)
        self.wait(2.0)

    def rebundle(self):
        n1, mid = N + 1, TOPW - 2 * CH
        leak = self.kids[LEAK]
        self.groups4 = [[b + (0,) for b in self.groups[y]] if y <= N else [] for y in range(n1 + 1)]
        for y in range(1, n1 + 1):
            self.groups4[y] += [b + (1,) for b in self.groups[y - 1]]
        lefts, spans = equal_layout(self.groups4, EW4, PG, BG2)
        spot = {m: lefts[y][k] + EW4 / 2 for y, g in enumerate(self.groups4) for k, m in enumerate(g)}

        self.caption(r"Sorting by the new count gathers the bundles of $n=4$")
        self.clear_slots()
        others = [m for m in self.kids if m != LEAK]
        self.play(*[self.kids[m].animate(path_arc=-0.6).move_to([spot[m], mid, 0]) for m in others],
                  *[self.klabels[m].animate(path_arc=-0.6).move_to([spot[m], TOPW + 0.3, 0]) for m in others],
                  FadeOut(self.wedge), FadeOut(self.factors), run_time=2.5)
        self.wait(0.6)

        self.caption(r"FFFS, the success child of FFF, moves to count 1")
        box = SurroundingRectangle(leak, color=INK, buff=0.05, stroke_width=2.5).set_z_index(3)
        self.play(Create(box), run_time=0.6)
        shift = np.array([spot[LEAK], mid, 0]) - leak.get_center()
        self.play(*[m.animate(path_arc=-1.2).shift(shift) for m in (leak, box, self.klabels[LEAK])], run_time=1.8)
        self.wait(0.6)

        ylab = VGroup(*[MathTex(str(y), font_size=26).move_to([(a + b) / 2, TOPW + 0.3, 0])
                        for y, (a, b) in enumerate(spans)])
        self.play(FadeOut(VGroup(*self.klabels.values())), FadeIn(ylab), run_time=0.8)
        self.caption(r"Count 1 fails at $n=4$, so FFFS has left the passing set")
        self.equation(r"\mathcal S(4)=\{0\}")
        self.wedge = edge_line(spans[0][1] + BG2 / 2, mid, n1 * CH)
        self.play(Create(self.wedge), *[Transform(k, repaint(k, m, colour(n1, sum(m)))) for m, k in self.kids.items()],
                  run_time=1.2)
        self.wait(2.0)

        self.caption(r"Draw each child with width equal to its probability")
        self.clear_slots()
        lefts, spans = lay([[seq_prob(n1, y) * W] * len(g) for y, g in enumerate(self.groups4)], 0.0, G, X0)
        anims = []
        for y, g in enumerate(self.groups4):
            w = seq_prob(n1, y) * W
            for k, m in enumerate(g):
                t = column(m, w, lefts[y][k], TOPW, colour(n1, y))
                anims.append(Transform(self.kids[m], t))
                if m == LEAK:
                    anims.append(Transform(box, SurroundingRectangle(t, color=INK, buff=0.05, stroke_width=2.5)
                                           .set_z_index(3)))
        ylab4 = row_labels(spans, TOPW + 0.3, [str(y) for y in range(n1 + 1)], 26, 0.3)
        self.play(*anims, FadeOut(ylab), self.wedge.animate.move_to([spans[0][1] + G / 2, mid, 0]), run_time=2.2)
        self.play(FadeIn(ylab4), run_time=0.5)
        self.wait(1.0)

        self.caption(r"Bundle 1 at $n=4$ holds FFFS and the failure children of bundle 1")
        self.equation(r"m_4(1)=m_3(1)\{1-p_3(1)\}+m_3(0)\,p_3(0)")
        self.wait(3.0)

        self.caption(r"Only FFFS crosses out of the passing set")
        self.clear_slots()
        loss = R.pmf[N][0] * p_s(N, 0)
        top0 = self.cols[(0,) * N]
        xs = top0.get_left()[0] + top0.width * (1 - p_s(N, 0))
        divider = DashedLine([xs, top0.get_bottom()[1], 0], [xs, top0.get_top()[1], 0], color=INK,
                             stroke_width=1.5, dash_length=0.06)
        my = self.mass_row[1][0].get_y() - 0.08
        shares = VGroup(MathTex(rf"\Pr(\mathrm{{FFFF}})={R.pmf[n1][0]:.3f}", font_size=22)
                        .move_to([(top0.get_left()[0] + xs) / 2, my, 0]),
                        MathTex(rf"\Pr(\mathrm{{FFFS}})={loss:.3f}", font_size=22)
                        .move_to([(xs + top0.get_right()[0]) / 2 - 0.1, my, 0]))
        self.play(Create(divider), FadeOut(self.mass_row[1][0]), FadeIn(shares), run_time=1.0)
        trail = CurvedArrow(np.array([(xs + top0.get_right()[0]) / 2, my - 0.22, 0]), leak.get_top() + 0.1 * UP,
                            angle=0.5, color=INK, stroke_width=2.5, tip_length=0.18)
        a0, b0 = spans[0]
        under = YB - n1 * CH / 2 - 0.12
        braces = VGroup(brace_under(a0, b0, under, rf"A(4)=m_4(0)={R.A[n1]:.3f}"),
                        brace_under(leak.get_left()[0], leak.get_right()[0], under,
                                    rf"m_3(0)\,p_3(0)={loss:.3f}"))
        self.play(Create(trail), run_time=1.0)
        self.play(FadeIn(braces), run_time=0.8)
        self.equation(r"A(4)=A(3)-m_3(0)\,p_3(0)")
        self.side_note(rf"${R.A[n1]:.3f}={R.A[N]:.3f}-{loss:.3f}$")
        self.wait(3.5)

        self.caption(r"Every other sequence stays on the side of the boundary its parent was on")
        self.wait(3.0)
        self.caption(r"With the edge at $j-1$, assurance loses only that bundle's success share")
        self.equation(r"A(n+1)=A(n)-m_n(j-1)\,p_n(j-1)")
        self.side_note(r"Here $j=1$ and the edge bundle is $y=0$")
        self.wait(4.0)
