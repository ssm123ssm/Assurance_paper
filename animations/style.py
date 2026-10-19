import os
import re

from manim import *

CAPTIONS = os.environ.get("CAPTIONS", "1") != "0"

BG = "#FFFFFF"
INK = "#1F2328"
MUTED = "#8A8F98"
NEUTRAL = "#A7ADB5"
PASS = "#0072B2"
FAIL = "#D55E00"
SUS = "#1A9E77"

config.background_color = BG
for _cls in (Tex, MathTex, Text, DecimalNumber, Integer):
    _cls.set_default(color=INK)


class Narrated:

    eq_center = ORIGIN
    note_center = ORIGIN
    slot_width = 4.9

    def init_slots(self):
        self.cap = self.eq = self.note = None

    def caption(self, text, run_time=0.6):
        words = re.sub(r"\$[^$]*\$", "", text)
        self.next_section(name="_".join(re.findall(r"[A-Za-z]+", words)[:4]).lower() or "beat")
        if not CAPTIONS:
            return
        new = Tex(text, font_size=32).to_edge(UP, buff=0.3)
        anims = [FadeIn(new, shift=0.1 * DOWN)]
        if self.cap is not None:
            anims.append(FadeOut(self.cap, shift=0.1 * UP))
        self.play(*anims, run_time=run_time)
        self.cap = new

    def _slot(self, attr, mob, center, run_time):
        if mob.width > self.slot_width:
            mob.scale_to_fit_width(self.slot_width)
        mob.move_to(center)
        old = getattr(self, attr)
        anims = [FadeIn(mob)] + ([FadeOut(old)] if old is not None else [])
        self.play(*anims, run_time=run_time)
        setattr(self, attr, mob)

    def equation(self, tex, run_time=0.6, font_size=32):
        self._slot("eq", MathTex(tex, font_size=font_size), self.eq_center, run_time)

    def side_note(self, text, run_time=0.5):
        self._slot("note", Tex(text, font_size=26, color=MUTED), self.note_center, run_time)

    def clear_slots(self, run_time=0.4):
        olds = [m for m in (self.eq, self.note) if m is not None]
        if olds:
            self.play(*[FadeOut(m) for m in olds], run_time=run_time)
        self.eq = self.note = None
