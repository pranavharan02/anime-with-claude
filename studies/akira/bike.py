"""The red bike and its rider, side profile facing right, as a night cel.

Original design in the spirit of the film's capsule bike. Drawn in a local
1000 x 600 box (ground at y = 585) and placed with a canvas transform.
"""
import numpy as np
import skia

from engine.core import col
from engine.ink import Pen, curve, path_of, INK

# night cel palette: every colour already repainted for the night key
RED, RED_SH, RED_HI, RED_SPEC = '#c0172f', '#5e0f35', '#ff5a4f', '#ffc2b0'
DARK, DARK_SH = '#232b37', '#121720'
METAL, METAL_SH, METAL_HI = '#64737f', '#3a4450', '#b7d4dc'
NAVY, NAVY_SH = '#26304a', '#141a2c'
SKIN, SKIN_SH = '#c39a8c', '#7f5f66'
HAIR, HAIR_HI = '#0e0d14', '#3b4a63'
RIM = '#8ff0ec'      # cyan rim light from the city
GLASS = '#63b7c4'

REAR, FRONT = (205.0, 470.0), (745.0, 475.0)
R_TIRE, F_TIRE = 112.0, 108.0

SHELL = [(70, 455, 'c'), (44, 420), (40, 372), (62, 328), (120, 297), (200, 283), (275, 292),
         (325, 318, 'c'), (480, 322, 'c'), (540, 296), (620, 280), (710, 290), (800, 322),
         (865, 368), (892, 415), (884, 452, 'c'), (834, 451, 'c'), (804, 405), (745, 383),
         (686, 405), (656, 451, 'c'), (612, 452, 'c'), (540, 436), (430, 440), (340, 452, 'c'),
         (297, 445, 'c'), (266, 397), (205, 375), (144, 397), (113, 445, 'c')]


def wheel(pen, c, r, spin, moving=True):
    cx, cy = c
    disc = lambda rr: [(cx + rr * np.cos(a), cy + rr * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)]
    pen.fill(disc(r), DARK)
    # tyre shadow (lower-left crescent) and rim light (top arc)
    with pen.clip(disc(r)):
        shadow = [(x - r * 0.12, y + r * 0.14) for x, y in disc(r * 1.1)]
        pen.fill(shadow, DARK_SH)
    pen.fill(disc(r * 0.74), METAL_SH)
    pen.fill(disc(r * 0.62), METAL)
    with pen.clip(disc(r * 0.62)):
        pen.fill([(cx - r, cy + r * 0.05), (cx + r, cy - r * 0.25), (cx + r, cy + r), (cx - r, cy + r)], METAL_SH, smooth=False)
    pen.fill(disc(r * 0.2), METAL_HI)
    pen.fill(disc(r * 0.09), DARK)
    # spin: blurred spoke arcs when moving, crisp bolts when still
    if moving:
        for k in range(5):
            a0 = spin + k * 2 * np.pi / 5
            arc = [(cx + r * 0.42 * np.cos(a), cy + r * 0.42 * np.sin(a)) for a in np.linspace(a0, a0 + 0.9, 8)]
            pen.line(arc, w=3.0, color=METAL_HI, taper=0.5, alpha=0.55, wobble=0.2)
    pen.line(disc(r), w=2.3, closed=True)
    pen.line(disc(r * 0.74), w=1.6, closed=True)
    pen.line(disc(r * 0.2), w=1.4, closed=True)
    arc = [(cx + (r - 3) * np.cos(a), cy + (r - 3) * np.sin(a)) for a in np.linspace(-2.6, -0.9, 10)]
    pen.line(arc, w=2.2, color=RIM, taper=0.4, wobble=0.15)


def body(pen):
    # engine block and underside, behind the shell
    pen.fill([(318, 440), (430, 430), (560, 428), (640, 440), (620, 540, 'c'), (520, 562), (380, 556), (318, 520, 'c')], DARK)
    pen.fill([(345, 520), (520, 540), (610, 515), (600, 548, 'c'), (380, 552, 'c')], DARK_SH)
    pen.fill([(360, 470), (600, 462), (606, 486), (366, 494)], METAL_SH)
    pen.fill([(366, 472), (598, 465), (600, 472), (368, 480)], METAL_HI, alpha=0.7)
    pen.line([(360, 470), (600, 462), (606, 486), (366, 494)], w=1.5, closed=True, smooth=False)
    for x in (380, 420, 460, 500, 540):  # cooling fins
        pen.line([(x, 516), (x + 10, 548)], w=1.4, taper=0.3)
    pen.line([(318, 440), (430, 430), (560, 428), (640, 440), (620, 540, 'c'), (520, 562), (380, 556), (318, 520, 'c')],
             w=2.0, closed=True)

    poly = pen.fill(SHELL, RED)
    with pen.clip(SHELL):
        pen.fill([(-10, 398), (150, 352), (320, 372), (520, 362), (700, 345), (910, 400), (950, 650), (-10, 650)], RED_SH)
        # long highlight band along the spine, then a hard specular streak
        pen.fill([(60, 330), (130, 302), (230, 290), (290, 302), (300, 312), (220, 302), (130, 312), (70, 345)], RED_HI)
        pen.fill([(540, 300), (630, 285), (720, 296), (810, 330), (860, 372), (850, 380), (790, 342), (700, 310), (620, 300), (548, 310)], RED_HI)
        pen.fill([(600, 292), (690, 296), (760, 316), (754, 322), (690, 304), (604, 300)], RED_SPEC)
        pen.fill([(90, 318), (160, 300), (200, 298), (150, 308), (96, 326)], RED_SPEC)
    # panel seams
    pen.line([(480, 330), (500, 400), (560, 452), (640, 446)], w=1.5, taper=0.25)
    pen.line([(330, 330), (320, 380), (300, 420)], w=1.4, taper=0.3)
    pen.line([(780, 350), (820, 395), (840, 440)], w=1.3, taper=0.3)
    # tail light housing and headlight
    pen.fill([(40, 370), (50, 352), (62, 352), (56, 398), (44, 400)], '#ff4a2e')
    pen.fill([(866, 392), (888, 404), (890, 424), (872, 420)], '#fff2c8')
    pen.line(SHELL, w=2.4, closed=True)
    # seat
    seat = [(326, 318, 'c'), (338, 304), (400, 300), (470, 304), (482, 322, 'c')]
    pen.fill(seat, DARK)
    pen.line(seat, w=1.8, closed=True)
    # windscreen bubble: tinted glass with a reflection streak
    screen = [(612, 283, 'c'), (650, 250), (712, 244), (770, 270), (800, 322, 'c'), (710, 291), (640, 281)]
    pen.fill(screen, GLASS, alpha=0.45)
    pen.fill([(660, 258), (720, 252), (760, 272), (718, 262), (668, 266)], '#e9ffff', alpha=0.8)
    pen.line(screen, w=1.8, closed=True)
    # rim light along the top of the shell
    pen.line([(66, 330), (120, 300), (200, 287), (270, 296)], w=2.4, color=RIM, taper=0.35)
    pen.line([(545, 298), (620, 284)], w=2.0, color=RIM, taper=0.4)
    pen.line([(805, 326), (862, 370), (886, 410)], w=2.0, color=RIM, taper=0.35)


def rider(pen, bob=0.0):
    c = pen.c
    c.save()
    c.translate(0, bob)
    # far arm (behind the torso), darker
    sleeve_far = [(560, 215), (612, 240), (660, 256), (690, 258), (692, 272), (650, 274), (600, 258), (548, 236)]
    pen.fill(sleeve_far, RED_SH)
    pen.line(sleeve_far, w=1.6, closed=True)

    # leg: thigh along the tank, shin tucked back to the peg
    thigh = [(372, 300), (440, 290), (520, 300), (560, 318), (556, 342), (520, 348), (440, 334), (384, 322)]
    pen.fill(thigh, NAVY)
    with pen.clip(thigh):
        pen.fill([(360, 322), (450, 318), (560, 330), (560, 360), (360, 360)], NAVY_SH)
    pen.line(thigh, w=2.0, closed=True)
    shin = [(520, 330), (556, 336), (548, 360), (500, 418), (470, 410), (510, 352)]
    pen.fill(shin, NAVY_SH)
    pen.line(shin, w=2.0, closed=True)
    boot = [(468, 404), (504, 412), (522, 428), (520, 440, 'c'), (452, 440, 'c'), (450, 420)]
    pen.fill(boot, DARK)
    pen.line(boot, w=2.0, closed=True)

    # torso: jacket hunched forward over the tank
    torso = [(350, 304, 'c'), (356, 262), (392, 226), (452, 198), (520, 180), (574, 178), (608, 192),
             (612, 218), (590, 240), (540, 262), (480, 296), (452, 306, 'c')]
    pen.fill(torso, RED)
    with pen.clip(torso):
        pen.fill([(330, 268), (420, 250), (520, 230), (600, 222), (640, 260), (600, 330), (330, 330)], RED_SH)
        pen.fill([(380, 236), (450, 204), (520, 186), (570, 184), (520, 196), (450, 216), (392, 246)], RED_HI)
    pen.line(torso, w=2.3, closed=True)
    pen.line([(372, 284), (420, 262), (470, 250)], w=1.3, taper=0.4)  # jacket crease
    pen.line([(380, 236), (450, 202), (520, 184), (572, 182)], w=2.2, color=RIM, taper=0.3)
    # belt line where the jacket meets the pants
    pen.line([(352, 300), (452, 304)], w=1.6, taper=0.2)

    # head, low behind the screen
    head = [(586, 196), (590, 168), (612, 150), (640, 146), (658, 156), (666, 172), (662, 180),
            (668, 190), (660, 194), (654, 206), (636, 212), (612, 210)]
    pen.fill(head, SKIN)
    with pen.clip(head):
        pen.fill([(570, 150), (630, 150), (628, 190), (640, 216), (570, 216)], SKIN_SH)
    pen.line(head, w=2.1, closed=True)
    # hair, swept back by the wind
    hair = [(588, 182, 'c'), (586, 160), (560, 150, 'c'), (590, 146), (572, 132, 'c'), (606, 134),
            (600, 118, 'c'), (630, 130), (652, 136), (666, 150), (654, 152), (630, 150), (612, 160), (604, 180, 'c')]
    pen.fill(hair, HAIR)
    pen.line([(596, 136), (624, 132), (650, 140)], w=2.0, color=HAIR_HI, taper=0.4)
    pen.line(hair, w=1.8, closed=True)
    # goggles pushed up on the forehead
    gog = [(618, 150), (650, 146), (664, 154), (660, 164), (628, 166), (616, 160)]
    pen.fill(gog, '#3c3a46')
    pen.fill([(640, 151), (658, 152), (660, 160), (642, 160)], '#e8b25c')
    pen.line(gog, w=1.6, closed=True)
    # eye and brow, profile
    pen.line([(648, 172), (660, 170)], w=2.2, taper=0.2)
    pen.line([(646, 166), (662, 164)], w=2.6, taper=0.3)
    # near arm reaching for the bars
    sleeve = [(556, 200), (598, 206), (646, 236), (690, 252), (702, 262), (694, 276), (650, 270),
              (600, 248), (556, 230)]
    pen.fill(sleeve, RED)
    with pen.clip(sleeve):
        pen.fill([(540, 226), (600, 240), (700, 264), (700, 290), (540, 290)], RED_SH)
    pen.line(sleeve, w=2.2, closed=True)
    pen.line([(560, 202), (600, 208), (646, 236)], w=2.0, color=RIM, taper=0.3)
    glove = [(690, 252), (712, 256), (722, 266), (712, 278), (694, 276)]
    pen.fill(glove, DARK)
    pen.line(glove, w=1.8, closed=True)
    c.restore()


def draw(canvas, emit, x, y, scale, spin=0.0, bob=0.0, seed=0, moving=True):
    """Draw bike + rider with the local origin mapped to (x, y)."""
    for cv in (canvas, emit):
        cv.save()
        cv.translate(x, y)
        cv.scale(scale, scale)
    pen = Pen(canvas, seed)
    wheel(pen, REAR, R_TIRE, spin, moving)
    wheel(pen, FRONT, F_TIRE, spin * 1.04, moving)
    body(pen)
    rider(pen, bob)
    # lamps into the emission pass
    emit.drawCircle(46, 380, 16, skia.Paint(AntiAlias=True, Color4f=col('#ff3a20', 1.0)))
    emit.drawCircle(880, 410, 14, skia.Paint(AntiAlias=True, Color4f=col('#fff0c0', 1.0)))
    for cv in (canvas, emit):
        cv.restore()
    return (x + 46 * scale, y + 380 * scale)  # tail light position on screen
