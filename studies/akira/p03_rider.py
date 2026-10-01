"""Plate 03: the rider, close-up. Character cel test.

Three-quarter view facing screen left. Night lighting: a warm sodium key from
the upper left, a cyan rim from the city on the right. Two tones per material
with hard shadow edges; hair and jacket get a third highlight tone.
Face authored in a 600 x 700 local box.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import skia

from engine.core import W, H, surface, to_np, flatten, save, col, hx
from engine.city import rect
from engine.fx import bloom, lin_grad, blur
from engine.ink import Pen, path_of, INK
from engine.film import develop

OUT = ROOT / 'out' / 'akira'

SKIN, SKIN_SH, SKIN_HI = '#d99c7e', '#7d5671', '#f2c3a2'
LIP = '#b0706c'
HAIR, HAIR_SH, HAIR_HI = '#17121c', '#0a080e', '#3a4462'
RED, RED_SH, RED_HI = '#c42032', '#5a1236', '#ff6a52'
SHIRT, SHIRT_SH = '#d9d4e2', '#8b86a8'
EYE_W, IRIS, PUPIL = '#efe2da', '#5a3326', '#140c0c'
RIM = '#9ff4ef'
LINE = '#22131a'

FACE = [(176, 150), (156, 214), (160, 252), (166, 268), (154, 298), (150, 330), (162, 360),
        (176, 398), (198, 440), (222, 474), (248, 496), (282, 502), (330, 482), (382, 440, 'c'),
        (404, 400), (420, 330), (430, 250), (420, 180), (360, 130), (260, 120)]
NECK = [(282, 482), (392, 436), (410, 470), (420, 640, 'c'), (284, 640, 'c'), (290, 540)]
CAP = [(172, 192), (160, 130), (200, 70), (280, 38), (370, 42), (450, 90), (486, 170), (484, 262),
       (460, 350), (432, 300), (420, 240), (380, 204), (300, 186), (220, 188)]
# (base a, base b, tip, bulge): bangs, crown spikes swept back, nape
BANGS = [((176, 168), (228, 178), (146, 252), 6), ((212, 176), (272, 184), (206, 266), 7),
         ((256, 180), (322, 188), (266, 254), 7), ((306, 186), (362, 198), (334, 262), 6),
         ((350, 194), (404, 214), (384, 272), 5)]
CROWN = [((168, 150), (222, 92), (112, 104), 8), ((202, 84), (272, 50), (176, 4), 9),
         ((258, 46), (342, 36), (290, -26), 10), ((330, 38), (412, 60), (424, -10), 10),
         ((398, 60), (462, 110), (520, 34), 9), ((448, 100), (486, 182), (572, 128), 8),
         ((480, 170), (490, 262), (566, 252), 8), ((486, 250), (470, 340), (536, 374), 7),
         ((468, 316), (440, 380), (470, 446), 6), ((418, 226), (452, 246), (430, 352), 5)]


def clump(a, b, tip, bulge):
    """One lock of hair: two curved sides meeting in a sharp tip."""
    a, b, tip = (np.array(v, float) for v in (a, b, tip))
    base = (a + b) / 2
    axis = tip - base
    n = np.array([-axis[1], axis[0]]) / (np.linalg.norm(axis) + 1e-9)
    side = 1 if np.dot(a - base, n) > 0 else -1
    ma = a + (tip - a) * 0.45 + n * bulge * side
    mb = b + (tip - b) * 0.45 - n * bulge * side
    return [tuple(a), tuple(ma), (tip[0], tip[1], 'c'), tuple(mb), tuple(b)]


def neck(pen):
    pen.fill(NECK, SKIN)
    with pen.clip(NECK):
        pen.fill([(250, 440), (430, 420), (440, 560), (330, 545), (262, 520)], SKIN_SH)
    pen.line([(290, 520), (284, 640)], w=2.2, taper=0.3)
    pen.line([(410, 470), (420, 640)], w=2.2, taper=0.3)
    pen.line([(412, 480), (420, 600)], w=2.6, color=RIM, taper=0.4)


def face(pen):
    # ear (near side, partly behind the hair)
    ear = [(404, 284), (432, 268), (452, 290), (448, 340), (428, 378), (408, 380)]
    pen.fill(ear, SKIN)
    with pen.clip(ear):
        pen.fill([(420, 260), (470, 260), (470, 390), (430, 390)], SKIN_SH)
    pen.line(ear, w=2.0, closed=True)
    pen.line([(428, 296), (440, 318), (430, 350)], w=1.5, taper=0.4)

    pen.fill(FACE, SKIN)
    with pen.clip(FACE):
        # the cheek plane turning away from the key light
        pen.fill([(336, 250), (372, 300), (380, 380), (350, 452), (300, 498), (420, 520), (450, 200), (360, 200)], SKIN_SH)
        # cast shadow of the hair on the forehead
        for a, b, tip, bl in BANGS:
            sh = [(x + 12, y + 14) for x, y, *_ in clump(a, b, tip, bl + 3)]
            pen.fill(sh, SKIN_SH)
        pen.fill([(150, 150), (420, 150), (420, 206), (150, 206)], SKIN_SH)
        # eye sockets under the brow
        pen.fill([(236, 268), (270, 262), (330, 262), (350, 276), (320, 272), (262, 274)], SKIN_SH)
        pen.fill([(160, 272), (196, 268), (214, 278), (190, 278), (166, 284)], SKIN_SH)
        # nose: shadow side and the cast shadow under it
        pen.fill([(216, 298), (221, 334), (208, 366), (224, 376), (234, 362), (228, 322)], SKIN_SH)
        # under the lower lip and the chin's underside
        pen.fill([(222, 446), (250, 442), (272, 448), (252, 456)], SKIN_SH)
        pen.fill([(236, 488), (282, 492), (330, 474), (330, 520), (230, 520)], SKIN_SH)
        # a warm highlight on the lit cheekbone and nose bridge
        pen.fill([(168, 306), (190, 300), (198, 318), (176, 328)], SKIN_HI)
        pen.fill([(206, 300), (212, 300), (208, 340), (202, 346)], SKIN_HI)
    pen.line(FACE[:14], w=2.5)  # front contour (open: the back is under the hair)

    # nose line, nostril, mouth (Otomo: a real nose, a wide mouth)
    pen.line([(214, 288), (204, 330), (186, 364), (196, 372), (214, 370)], w=1.9, taper=0.3)
    pen.line([(226, 366), (234, 372)], w=1.8, taper=0.2)
    pen.line([(204, 420), (226, 418), (250, 424), (282, 418), (292, 410)], w=2.3, taper=0.25)
    pen.line([(232, 438), (252, 440)], w=1.5, taper=0.4, color=LIP)
    # cheek scar, a little history
    pen.line([(330, 378), (348, 368)], w=1.4, taper=0.4)
    pen.line([(334, 386), (346, 380)], w=1.0, taper=0.4)

    # eyes: near (larger) and far (foreshortened)
    near = [(250, 296), (268, 284), (300, 280), (326, 288), (336, 300, 'c'), (310, 306), (276, 308), (256, 304)]
    pen.fill(near, EYE_W)
    with pen.clip(near):
        pen.fill([(260, 278), (292, 278), (292, 316), (260, 316)], EYE_W)
        iris = [(281 + 15 * np.cos(a), 295 + 16 * np.sin(a)) for a in np.linspace(0, 6.28, 20, endpoint=False)]
        pen.fill(iris, IRIS)
        pen.fill([(281 + 8 * np.cos(a), 295 + 8 * np.sin(a)) for a in np.linspace(0, 6.28, 14, endpoint=False)], PUPIL)
        pen.fill([(240, 270), (340, 270), (340, 290), (240, 292)], SKIN_SH, alpha=0.55)  # lid shadow
        pen.fill([(286 + 3.5 * np.cos(a), 289 + 3.5 * np.sin(a)) for a in np.linspace(0, 6.28, 10, endpoint=False)], '#ffffff')
    pen.line([(246, 298), (262, 284), (300, 278), (326, 286), (340, 302)], w=3.6, taper=0.25)  # heavy upper lid
    pen.line([(262, 308), (290, 310), (318, 304)], w=1.3, taper=0.4)
    far = [(176, 296), (190, 286), (210, 286), (220, 296, 'c'), (204, 302), (184, 302)]
    pen.fill(far, EYE_W)
    with pen.clip(far):
        pen.fill([(196 + 9 * np.cos(a), 294 + 12 * np.sin(a)) for a in np.linspace(0, 6.28, 16, endpoint=False)], IRIS)
        pen.fill([(196 + 5 * np.cos(a), 294 + 6 * np.sin(a)) for a in np.linspace(0, 6.28, 12, endpoint=False)], PUPIL)
        pen.fill([(198 + 2.5 * np.cos(a), 290 + 2.5 * np.sin(a)) for a in np.linspace(0, 6.28, 8, endpoint=False)], '#ffffff')
    pen.line([(172, 298), (190, 284), (212, 284), (224, 296)], w=3.0, taper=0.25)
    # brows: thick, low, angled in
    pen.fill([(246, 262), (282, 248), (322, 244), (348, 252), (322, 252), (284, 258), (250, 268)], HAIR)
    pen.fill([(166, 262), (188, 254), (214, 258), (190, 262), (170, 268)], HAIR)

    # hair over everything on the head: cap + locks, inked as one silhouette
    locks = [clump(*c) for c in BANGS + CROWN]
    shapes = [CAP] + locks
    for sh in shapes:
        pen.fill(sh, HAIR)
    outline = pen.union_outline(shapes)
    with pen.clip(outline[0], smooth=False):
        pen.fill([(392, -40), (620, -40), (620, 480), (420, 480), (436, 240)], HAIR_SH)
        # light catching the crown, one stroke per lock
        for a, b, tip, _ in CROWN[:6]:
            base = (np.add(a, b)) / 2
            t = np.array(tip, float) - base
            p0, p1 = base + t * 0.18, base + t * 0.62
            off = np.array([-8, -4])
            pen.line([tuple(p0 + off), tuple((p0 + p1) / 2 + off * 1.4), tuple(p1 + off)],
                     w=11, color=HAIR_HI, taper=0.55, wobble=0.6, alpha=0.85)
    for poly in outline:
        pen.line(poly, w=2.5, closed=True, smooth=False)
    # strand lines running from tips back into the mass
    for a, b, tip, _ in BANGS + CROWN[1:7]:
        base = (np.add(a, b)) / 2
        t = np.array(tip, float) - base
        pen.line([tuple(np.array(tip) - t * 0.02), tuple(base + t * 0.35 + [3, 2])], w=1.5, taper=0.6)

    # cyan rim from the city on the back of the hair, the jaw and the neck
    # cyan rim: trace the back of the hair silhouette, just inside the line
    back = outline[0]
    cx = back[:, 0].mean()
    run = [tuple(p + [-4, 0]) for p in back if p[0] > 470 and 40 < p[1] < 420]
    if len(run) > 2:
        run.sort(key=lambda q: q[1])
        pen.line(run, w=3.0, color=RIM, taper=0.3, smooth=False)
    pen.line([(388, 438), (406, 400)], w=2.2, color=RIM, taper=0.4)


def jacket(pen):
    # body of the jacket, shoulders run off frame
    body = [(-200, 760, 'c'), (-120, 640), (40, 580), (200, 548), (300, 590), (360, 600), (440, 548),
            (600, 570), (760, 640), (860, 760, 'c'), (860, 900, 'c'), (-200, 900, 'c')]
    pen.fill(body, RED)
    with pen.clip(body):
        pen.fill([(380, 560), (520, 600), (700, 640), (900, 700), (900, 900), (420, 900), (380, 700)], RED_SH)
        pen.fill([(-120, 650), (40, 592), (180, 566), (120, 600), (0, 640), (-100, 690)], RED_HI)
        pen.fill([(-200, 820), (120, 780), (300, 800), (300, 900), (-200, 900)], RED_SH, alpha=0.6)
    pen.line(body[:11], w=2.6)
    # undershirt in the V
    v = [(266, 560), (330, 640), (400, 556), (380, 700, 'c'), (290, 700, 'c')]
    pen.fill(v, SHIRT)
    with pen.clip(v):
        pen.fill([(340, 560), (420, 556), (400, 720), (350, 720)], SHIRT_SH)
    pen.line([(266, 560), (330, 640), (400, 556)], w=2.0)
    # stand-up collar, both sides
    lc = [(196, 470), (262, 500), (276, 560), (330, 650), (250, 600), (190, 560)]
    rc = [(404, 452), (450, 430), (470, 520), (420, 600), (330, 650), (400, 560)]
    for shape, base, sh in ((lc, RED, RED_SH), (rc, RED_SH, '#3e0c28')):
        pen.fill(shape, base)
        pen.line(shape, w=2.4, closed=True)
    pen.line([(206, 478), (196, 540)], w=2.4, color=RED_HI, taper=0.3)
    pen.line([(462, 440), (474, 520)], w=2.6, color=RIM, taper=0.35)
    # seam and zip
    pen.line([(80, 600), (120, 700), (140, 800)], w=1.5, taper=0.3)
    pen.line([(560, 590), (600, 700)], w=1.5, taper=0.3)


def background(c, e, rs):
    lin_grad(c, path_of(rect(0, 0, W, H)), (0, 0), (0, H), ['#06111d', '#0e2a36', '#123943'], [0, 0.6, 1])
    # out-of-focus city lights: soft discs, painted and emitted
    for _ in range(70):
        x, y = rs.uniform(0, W), rs.uniform(0, H * 0.85)
        r = rs.uniform(10, 46)
        k = rs.choice(['#ffcf7a', '#ffe3a8', '#7ff0ff', '#ff6fb0', '#ff9d5c'], p=[.35, .2, .2, .1, .15])
        a = rs.uniform(0.12, 0.4)
        c.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Color4f=col(k, a)))
        e.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Color4f=col(k, a * 0.4)))
    # a vertical neon sign, far behind, out of focus
    c.drawRect(skia.Rect(260, 120, 300, 620), skia.Paint(AntiAlias=True, Color4f=col('#ff4f9a', 0.55)))
    e.drawRect(skia.Rect(260, 120, 300, 620), skia.Paint(AntiAlias=True, Color4f=col('#ff4f9a', 0.6)))


def render(seed=3):
    rs = np.random.default_rng(seed)
    bs, be = surface(), surface()
    background(bs.getCanvas(), be.getCanvas(), rs)
    bg = blur(flatten(to_np(bs)), 9)
    emit = blur(to_np(be)[..., :3], 9)

    cs = surface()
    c = cs.getCanvas()
    c.translate(800, 120)
    c.scale(1.34, 1.34)
    pen = Pen(c, seed)
    neck(pen)
    jacket(pen)
    face(pen)
    cel = to_np(cs)
    img = flatten(cel, bg)
    # sodium key spill: warm wash from the upper left over the whole frame
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spill = np.clip(1 - np.hypot(xx - 300, yy + 200) / 1900, 0, 1) ** 2
    img += spill[..., None] * hx('#ff9a4a') * 0.10
    img += bloom(emit, ((10, .4), (40, .5)))
    return img


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    img = render()
    save(develop(img, frame=11, seed=3), OUT / 'p03_rider.png')
    print('wrote', OUT / 'p03_rider.png')
