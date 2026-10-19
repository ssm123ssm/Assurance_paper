import numpy as np
from manim import *

from style import *


def rgb(color):
    return np.array(ManimColor(color).to_rgb())


class StackView(VGroup):
    EPS = 0.004

    def __init__(self, x, d, mass, colors, x_lim, d_lim, centre, x_ticks, d_ticks, fx=0.74, fd=0.62,
                 x_label="y", d_label="n", **camera):
        super().__init__()
        self.FX, self.FD = fx, fd
        self.k = len(x)
        self.bx = np.array(x, dtype=float)
        self.bd = np.array(d, dtype=float)
        self.bz = np.zeros(self.k)
        self.bm = np.array(mass, dtype=float)
        self.col = np.array([rgb(c) for c in colors])
        self.op = np.ones(self.k)
        self.x_lim, self.d_lim, self.centre = np.array(x_lim, float), np.array(d_lim, float), centre
        self.x_ticks, self.d_ticks = x_ticks, d_ticks

        self.psi, self.elev = 0.0, 0.0
        self.ux, self.ud, self.hscale = 0.3, 0.2, 2.0
        self.ox, self.oy = 0.0, 0.0
        self.floor_alpha, self.count_alpha, self.depth_alpha = 1.0, 1.0, 0.0
        self.num_alpha = 1.0
        for key, value in camera.items():
            setattr(self, key, value)

        self.floor, self.bars = VGroup(), VGroup()
        self.x_nums = [(t, MathTex(str(t), font_size=22)) for t in x_ticks]
        self.d_nums = [(t, MathTex(str(t), font_size=22)) for t in d_ticks]
        self.x_lab = MathTex(x_label, font_size=28)
        self.d_lab = MathTex(d_label, font_size=28)
        self.boards = VGroup(*[m for _, m in self.x_nums + self.d_nums], self.x_lab, self.d_lab)
        self.add(self.floor, self.bars, self.boards)
        self.add_updater(lambda m: m.redraw())
        self.redraw()

    def view_coords(self, x, d, z):
        x, d, z = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in (x, d, z)])
        X = (x - self.centre[0]) * self.ux
        Y = (d - self.centre[1]) * self.ud
        Z = z * self.hscale
        cp, sp = np.cos(self.psi), np.sin(self.psi)
        sx = X * cp + Y * sp
        far = -X * sp + Y * cp
        sy = Z * np.cos(self.elev) + far * np.sin(self.elev)
        return sx, sy, far

    def point(self, x, d, z=0.0):
        sx, sy, _ = self.view_coords(x, d, z)
        return np.stack([self.ox + sx, self.oy + sy, np.zeros_like(sx)], -1)

    def faces(self, i):
        hx, hd = self.FX / 2, self.FD / 2
        x0, x1 = self.bx[i] - hx, self.bx[i] + hx
        d0, d1 = self.bd[i] - hd, self.bd[i] + hd
        z0, z1 = self.bz[i], self.bz[i] + self.bm[i]
        xs = x1 if self.psi >= 0 else x0
        top = self.point([x0, x1, x1, x0], [d0, d0, d1, d1], z1)
        front = self.point([x0, x1, x1, x0], [d0, d0, d0, d0], [z0, z0, z1, z1])
        side = self.point([xs, xs, xs, xs], [d0, d1, d1, d0], [z0, z0, z1, z1])
        return top, front, side

    def shown(self, t, lim):
        return lim[0] - 1e-9 <= t <= lim[1] + 1e-9

    def redraw(self):
        self.floor.submobjects = self.draw_floor().submobjects
        self.bars.submobjects = self.draw_bars().submobjects
        self.place_boards()
        return self

    def draw_floor(self):
        (x0, x1), (d0, d1) = self.x_lim, self.d_lim
        fa, ca, da = self.floor_alpha, self.count_alpha, self.depth_alpha
        c = self.point([x0, x1, x1, x0], [d0, d0, d1, d1])
        g = VGroup()
        if fa * da > 0.01:
            g.add(Polygon(*c, stroke_width=0).set_fill(MUTED, opacity=0.07 * fa * da))
            g.add(Polygon(*c, stroke_color=MUTED, stroke_width=1.2, stroke_opacity=0.6 * fa * da)
                  .set_fill(opacity=0))
        if fa * ca > 0.01:
            g.add(Line(c[0], c[1], stroke_color=INK, stroke_width=2, stroke_opacity=fa * ca))
            for t in self.x_ticks:
                if self.shown(t, self.x_lim):
                    p = self.point(t, d0)
                    g.add(Line(p, p + 0.09 * DOWN, stroke_color=INK, stroke_width=2, stroke_opacity=fa * ca))
        if fa * da > 0.01:
            g.add(Line(c[1], c[2], stroke_color=INK, stroke_width=2, stroke_opacity=fa * da))
            for t in self.d_ticks:
                p = self.point(x1, t)
                g.add(Line(p, p + 0.09 * RIGHT, stroke_color=INK, stroke_width=2, stroke_opacity=fa * da))
        return g

    def draw_bars(self):
        f = float(np.clip(abs(self.psi) / (PI / 2), 0, 1))
        shade = (lambda c: c * 0.82 + 0.18,
                 lambda c: c * interpolate(1.0, 0.76, f),
                 lambda c: c * interpolate(0.76, 1.0, f))
        _, _, far = self.view_coords(self.bx, self.bd, 0)
        g = VGroup()
        for i in np.lexsort((self.bz, -far)):
            op = self.op[i] * np.clip(self.bm[i] / self.EPS, 0, 1)
            if op <= 0.01:
                continue
            for pts, s in zip(self.faces(i), shade):
                face = VMobject().set_points_as_corners([*pts, pts[0]])
                face.set_fill(ManimColor.from_rgb(np.clip(s(self.col[i]), 0, 1)), opacity=op)
                face.set_stroke(ManimColor.from_rgb(self.col[i] * 0.7), width=0.5, opacity=0.5 * op)
                g.add(face)
        return g

    def place_boards(self):
        (x0, x1), (d0, d1) = self.x_lim, self.d_lim
        fa = self.floor_alpha
        for t, m in self.x_nums:
            on = self.shown(t, self.x_lim)
            m.move_to(self.point(t, d0) + 0.32 * DOWN).set_opacity(fa * self.count_alpha * self.num_alpha * on)
        for t, m in self.d_nums:
            m.move_to(self.point(x1, t) + np.array([0.3, -0.1, 0])).set_opacity(fa * self.depth_alpha)
        self.x_lab.move_to(self.point(x1, d0) + np.array([0.35, -0.35, 0])).set_opacity(fa * self.count_alpha)
        self.d_lab.move_to(self.point(x1, d1) + np.array([0.4, 0.15, 0])).set_opacity(fa * self.depth_alpha)


class Tween(Animation):

    def __init__(self, view, lag=0.0, lag_key=None, **targets):
        self.view, self.lag, self.lag_key, self.targets = view, lag, lag_key, targets
        super().__init__(Mobject(), **{k: targets.pop(k) for k in ("run_time", "rate_func") if k in targets})

    def begin(self):
        self.start = {k: np.array(getattr(self.view, k), dtype=float) for k in self.targets}
        self.end = {k: np.broadcast_to(np.asarray(v, dtype=float), self.start[k].shape).copy()
                    for k, v in self.targets.items()}
        super().begin()

    def interpolate_mobject(self, alpha):
        for k in self.targets:
            s, e = self.start[k], self.end[k]
            if self.lag and s.ndim and s.shape[0] == self.view.k:
                local = np.clip((alpha - self.lag * self.lag_key) / (1 - self.lag), 0, 1)
                a = np.vectorize(self.rate_func, otypes=[float])(local).reshape(-1, *([1] * (s.ndim - 1)))
            else:
                a = self.rate_func(alpha)
            v = s + (e - s) * a
            setattr(self.view, k, float(v) if v.ndim == 0 else v)


class ViewScene(Narrated, Scene):

    def setup(self):
        self.init_slots()
        self.views = []

    def get_moving_and_static_mobjects(self, animations):
        moving, static = super().get_moving_and_static_mobjects(animations)
        parts = {id(m) for v in self.views for m in v.get_family()[1:]}
        return [m for m in moving if id(m) not in parts], static
