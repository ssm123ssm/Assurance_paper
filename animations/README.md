# Animations

Manim scenes for the companion site. Each scene follows one section of the paper and uses the paper's example.

## Section 2, `section2.py`

`Section2Assurance` runs about 74 seconds. The example is the one in Figure 1 of the paper, a Beta(2.5, 11) prior with 90% equal-tailed intervals, target width w* = 0.29 and required level q = 0.90.

1. One study of 11 observations. Predictive probabilities of each count, the posterior interval width of each count, the passing set {0, ..., 4} and A(11) = 0.911.
2. Every sample size from 0 to 20 as rows of 3D bars. The passing probability of each row stacks into one column, and a side view turns the columns into the assurance curve.
3. The falls in assurance, the first-crossing choice n = 11, the fall to A(12) = 0.886, sustained assurance and the sustained choice n = 13.

Every number on screen comes from `model.py`. Its last line checks that every count passes for 20 ≤ n ≤ 57, and Proposition 1 of the paper covers n ≥ 57, so the backward minimum over n ≤ 20 is the sustained assurance.

## Rendering

The scenes need Manim Community 0.21, a LaTeX installation, SciPy and ffmpeg.

```
./render.sh          # 1080p60, then copies ../videos/section2_assurance.mp4 and its poster
./render.sh -ql      # quick draft in media/
CAPTIONS=0 ./render.sh   # without the on-screen captions
```

`stack_view.py` draws the 3D bars with their own orthographic camera. It is adapted from the comprehensive exam animations. Colours follow Figure 1, with blue for passing counts and orange for failing counts.
