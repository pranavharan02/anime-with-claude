"""Original 01: "Glare". A new character, close-up, in Otomo's construction.

Measured from the Tetsuo close-up (frame 062): one flat skin tone with almost
no face shading; shadow only under the jaw, under the fringe, in the ears and
nostrils; heavy contours outside and thinner lines inside; small dark irises,
a double-lid crease, no nose bridge line. Palette muted like the print.
"""
import numpy as np

from kit import Frame, ellipse, ROOT

OUT = ROOT / 'out' / 'akira_originals'

SKIN, SKIN_SH = '#a48370', '#77595a'
HAIR, HAIR_HI = '#1d151c', '#4a4859'
SCLERA, IRIS, HIGHLIGHT = '#d6d9c8', '#24161a', '#eef0e2'
MOUTH, TEETH, TEETH_SH = '#3b1519', '#d8d4c1', '#a9a595'
RED, RED_SH, RED_DK = '#c8232c', '#8e0f19', '#5c050d'
PLASTER, PLASTER_SH = '#ddd0b6', '#b3a48c'
INK = '#1a060c'

FACE = [(600, 160), (588, 250), (594, 330), (604, 372), (614, 436), (640, 502), (678, 562, 'c'),
        (740, 620), (800, 662), (842, 676), (892, 676), (934, 660), (994, 614), (1052, 556, 'c'),
        (1084, 472), (1100, 392), (1108, 340), (1112, 270), (1110, 190), (1080, 120), (850, 96),
        (640, 112)]


def sky(f):
    rs = np.random.default_rng(7)

    def field(xx, yy):
        top, bot = np.array([196, 205, 190.]), np.array([223, 229, 213.])
        t = (yy / f.h)[..., None]
        base = top + (bot - top) * t
        # slow drifting smoke: a few soft blobs, darker and greener
        smoke = np.zeros_like(xx)
        for _ in range(9):
            cx, cy = rs.uniform(-100, 700), rs.uniform(-50, 600)
            r = rs.uniform(90, 240)
            smoke += np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.6)) ** 2)) * rs.uniform(0.3, 0.8)
        smoke = np.clip(smoke, 0, 1)[..., None]
        return base * (1 - smoke * 0.12) + np.array([150, 160, 145.]) * smoke * 0.12

    f.paint_field(field)


def jacket(f):
    neck = f.fill([(690, 560), (1012, 566), (1030, 718, 'c'), (676, 718, 'c')], SKIN)
    f.fill([(660, 560), (740, 630), (800, 672), (856, 690), (920, 680), (990, 635), (1040, 580),
            (1050, 720, 'c'), (660, 720, 'c')], SKIN_SH, inside=neck)
    f.ink([(690, 592), (684, 718)], w=3.2, taper=0.15)
    f.ink([(1010, 600), (1022, 718)], w=3.2, taper=0.15)
    left = [(520, 718, 'c'), (560, 610), (640, 560), (700, 590), (722, 660), (716, 718, 'c')]
    right = [(990, 718, 'c'), (1000, 640), (1040, 570), (1130, 590), (1200, 650), (1230, 718, 'c')]
    for shape, sh in ((left, [(640, 560), (700, 590), (722, 660), (716, 718), (650, 718), (620, 620)]),
                      (right, [(1130, 590), (1200, 650), (1230, 718), (1150, 718), (1120, 640)])):
        p = f.fill(shape, RED)
        f.fill(sh, RED_SH, inside=p)
        f.ink(shape, w=4.0, closed=True)
    f.ink([(560, 640), (600, 700)], w=2.0)                      # collar fold
    f.ink([(1150, 640), (1180, 700)], w=2.0)


def head(f):
    le = [(592, 262), (560, 246), (540, 290), (548, 352), (570, 402), (606, 414)]
    re = [(1110, 270), (1142, 258), (1158, 300), (1150, 362), (1124, 404), (1102, 398)]
    for e, inner in ((le, [(580, 290), (564, 300), (568, 350), (588, 380)]),
                     (re, [(1126, 292), (1140, 310), (1134, 352), (1118, 378)])):
        p = f.fill(e, SKIN)
        f.fill(inner + [(inner[-1][0] + 10, inner[-1][1] - 30), (inner[0][0] + 8, inner[0][1] + 8)],
               SKIN_SH, inside=p)
        f.ink(e, w=3.6)
        f.ink(inner, w=2.0)

    face = f.fill(FACE, SKIN)
    # Otomo keeps the face nearly flat: one side plane and the fringe's shadow
    f.fill([(570, 220), (590, 330), (606, 436), (640, 502), (678, 562), (712, 592), (676, 500),
            (646, 420), (628, 330), (618, 220)], SKIN_SH, inside=face)
    rs_ = np.random.default_rng(4)          # same locks as the hair below
    xs = [1160]
    while xs[-1] > 600:
        xs.append(xs[-1] - rs_.uniform(32, 64))
    sh = [(1150, 100), (1150, 240)]
    for i in range(len(xs) - 1):
        xr, xl = xs[i], xs[i + 1]
        vy = rs_.uniform(118, 160)
        tx, ty = xl + (xr - xl) * rs_.uniform(0.05, 0.3), rs_.uniform(196, 252)
        sh += [(xr + 8, vy + 40), (tx + 12, ty + 20, 'c')]
    sh += [(560, 260), (560, 100)]
    f.fill(sh, SKIN_SH, inside=face)
    f.ink(FACE[:20], w=4.6, taper=0.06)

    # eyes: bigger irises tucked under heavy lids, glaring to screen left
    for sc, lid, flick, low, crease, iris_c, r in (
            ([(660, 312), (690, 286), (738, 276), (784, 284), (800, 304, 'c'), (770, 328), (712, 332)],
             [(652, 316), (690, 284), (742, 274), (790, 282), (806, 304)], [(652, 316), (638, 324)],
             [(690, 340), (730, 343), (772, 334)], [(680, 276), (734, 266), (782, 274)], (712, 303), 15),
            ([(896, 306), (922, 280), (976, 272), (1028, 280), (1046, 300, 'c'), (1010, 326), (944, 330)],
             [(890, 306), (924, 278), (980, 268), (1030, 276), (1052, 300)], [(1052, 300), (1066, 306)],
             [(930, 338), (974, 340), (1016, 332)], [(916, 270), (970, 260), (1022, 268)], (940, 299), 16)):
        p = f.fill(sc, SCLERA)
        f.fill(ellipse(iris_c[0], iris_c[1], r * 0.92, r, 20), IRIS, inside=p)
        f.fill([(sc[0][0] - 20, sc[0][1] - 40), (sc[3][0] + 20, sc[3][1] - 40), (sc[3][0] + 20, sc[3][1] - 7),
                (sc[0][0] - 20, sc[0][1] - 9)], '#a39f90', inside=p, smooth=False)   # lid shadow
        f.ink([(sc[1][0] + 4, sc[1][1] + 2), (sc[2][0], sc[2][1] + 1), (sc[3][0] - 2, sc[3][1] + 3)], w=1.4, taper=0.4)
        f.fill(ellipse(iris_c[0] + 4, iris_c[1] - 4, 2.6, 2.6, 8), HIGHLIGHT)
        f.ink(lid, w=6.4, taper=0.12)
        f.ink(flick, w=3.0, taper=0.5)
        f.ink(low, w=1.9, taper=0.35)
        f.ink(crease, w=1.8, taper=0.4)
    # brows: heavy, knotted down toward the nose
    f.ink([(640, 266), (700, 254), (756, 262), (798, 282)], w=9.5, taper=0.28)
    f.ink([(900, 278), (944, 256), (1000, 248), (1062, 254)], w=9.5, taper=0.28)
    f.ink([(836, 280), (844, 256)], w=1.9, taper=0.4)
    f.ink([(866, 278), (860, 254)], w=1.9, taper=0.4)
    f.ink([(700, 350), (734, 356), (764, 350)], w=1.6, taper=0.45)
    f.ink([(940, 348), (978, 352), (1008, 344)], w=1.6, taper=0.45)

    # nose: a plaster across the bridge, then only the tip and nostrils
    pl = [(806, 330, 'c'), (898, 322, 'c'), (902, 352, 'c'), (810, 360, 'c')]
    p = f.fill(pl, PLASTER)
    f.fill([(856, 320), (910, 316), (910, 360), (858, 360)], PLASTER_SH, inside=p, smooth=False)
    f.ink(pl, w=2.4, closed=True)
    for x in (826, 842, 870, 884):
        f.fill(ellipse(x, 341 - (x - 826) * 0.08, 1.7, 1.7, 6), '#9a8a72')
    f.fill([(820, 400), (838, 392), (850, 402), (832, 410)], '#3b2326')
    f.fill([(866, 402), (882, 392), (896, 400), (882, 410)], '#3b2326')
    f.ink([(808, 384), (822, 404), (852, 414), (880, 410), (904, 388)], w=2.8, taper=0.3)
    f.ink([(812, 364), (806, 384)], w=1.7, taper=0.4)

    # mouth: a snarl, the right corner pulled up
    mo = [(760, 520), (800, 502), (850, 496), (902, 498), (944, 498), (968, 484, 'c'), (954, 520),
          (916, 546), (860, 556), (806, 552), (772, 538, 'c')]
    p = f.fill(mo, MOUTH)
    f.fill([(762, 518), (810, 506), (860, 502), (910, 504), (962, 490), (950, 522), (860, 528), (780, 528)],
           TEETH, inside=p)
    f.fill([(786, 534), (860, 536), (940, 528), (916, 546), (860, 556), (806, 550)], TEETH, inside=p)
    f.fill([(762, 518), (810, 506), (860, 502), (910, 504), (962, 490), (962, 482), (760, 500)],
           TEETH_SH, inside=p, smooth=False)
    for x, t, lo in ((800, 0, True), (836, 1, False), (868, 0, True), (906, -3, False), (934, -8, True)):
        f.ink([(x, 506 + t), (x - 2, 524 + t * 0.5)], w=1.6, taper=0.3)
        if lo:
            f.ink([(x + 8, 536), (x + 6, 548)], w=1.3, taper=0.3)
    f.ink(mo, w=3.6, closed=True)
    f.ink([(816, 584), (852, 588), (892, 582)], w=2.3, taper=0.35)
    f.ink([(772, 450), (750, 504)], w=2.0, taper=0.4)
    f.ink([(944, 436), (982, 478)], w=2.4, taper=0.4)          # the sneer side bites deeper
    f.ink([(836, 642), (874, 644)], w=1.6, taper=0.45)
    f.ink([(1012, 404), (1044, 446)], w=1.8, taper=0.3)
    f.ink([(1016, 432), (1034, 418)], w=1.4, taper=0.3)

    # hair: uneven locks swept left by the wind, ending around the brows
    rs = np.random.default_rng(4)
    xs = [1160]
    while xs[-1] > 600:
        xs.append(xs[-1] - rs.uniform(32, 64))
    fringe, tips = [], []
    for i in range(len(xs) - 1):
        xr, xl = xs[i], xs[i + 1]
        valley = (xr, rs.uniform(118, 160))
        tip = (xl + (xr - xl) * rs.uniform(0.05, 0.3), rs.uniform(196, 252))
        fringe += [valley, (tip[0], tip[1], 'c')]
        tips.append((tip, valley))
    hair = ([(552, 300, 'c'), (530, 200), (538, 100), (580, 20), (610, -20, 'c'), (1222, -20, 'c'),
             (1202, 70), (1184, 170), (1162, 290, 'c')] + fringe + [(586, 140), (566, 220)])
    p = f.fill(hair, HAIR)
    for hl in ([(620, 96), (672, 64), (730, 50), (690, 70), (640, 104)],
               [(760, 46), (820, 34), (880, 34), (830, 46), (774, 58)],
               [(906, 36), (966, 38), (1022, 52), (970, 50), (912, 48)],
               [(1052, 60), (1096, 80), (1126, 112), (1092, 96), (1048, 74)]):
        f.fill(hl, HAIR_HI, inside=p)
    f.ink(hair[:-1], w=4.0, taper=0.04)
    for (tx, ty), (vx, vy) in tips:      # strand lines run from each tip up into the mass
        f.ink([(tx + 4, ty - 14), (tx + (vx - tx) * 0.35 + 6, vy - 30), (vx - 6, vy - 70)],
              w=1.6, color='#000000', taper=0.55)
    return [t for t, _ in tips]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = Frame()
    sky(f)
    jacket(f)
    head(f)
    f.camera(1.32, 850, 330, 900, 300)      # push in: the film frames heads big
    import json
    (OUT / 'o01_glare.json').write_text(json.dumps(f.scene(), separators=(',', ':')), encoding='utf8')
    f.shoot(OUT / 'o01_glare.png', lens=0.8, grain=1.3, seed=1)
    print('wrote', OUT / 'o01_glare.png')


if __name__ == '__main__':
    main()
