"""Original 02: "Rooftop". Night city, a lone rider on the parapet.

Vocabulary measured from frame 027: near-black navy sky (#04050f), dark teal
facades (#0d1623 / #122636), lit windows in pale yellow (#e0e1a0 / #bbb663)
with interiors, a grimy magenta parapet (#5a2943), violet far city.
"""
import json

import numpy as np

from kit import Frame, ellipse, ROOT

OUT = ROOT / 'out' / 'akira_originals'
INK = '#12040a'


def sky(f):
    def field(xx, yy):
        top, low = np.array([4, 5, 15.]), np.array([26, 18, 44.])
        t = np.clip(yy / 520, 0, 1)[..., None] ** 1.6
        glow = np.exp(-(((xx - 420) / 260) ** 2 + ((yy - 520) / 160) ** 2))[..., None]
        return top + (low - top) * t + glow * np.array([70, 30, 70.])
    f.paint_field(field)


def far_city(f, rs):
    """The violet city glowing in the gap, painted soft."""
    x = 250
    while x < 640:
        w = rs.uniform(18, 46)
        h = rs.uniform(60, 220)
        base = 520
        c = ['#3d2a52', '#4e3563', '#5b3f6e'][rs.integers(3)]
        f.fill([(x, base, 'c'), (x, base - h, 'c'), (x + w, base - h, 'c'), (x + w, base, 'c')], c, smooth=False,
               feather=0.8)
        for _ in range(int(h / 9)):
            if rs.random() < 0.55:
                wx, wy = x + rs.uniform(2, w - 4), base - rs.uniform(4, h - 4)
                f.fill(ellipse(wx, wy, 1.3, 1.1, 6), ['#e9a7d4', '#f1d7a0', '#c88be0'][rs.integers(3)])
        x += w * rs.uniform(0.7, 1.05)
    # a tall landmark tower with a lit crown
    f.fill([(430, 520, 'c'), (436, 250, 'c'), (470, 230, 'c'), (504, 250, 'c'), (510, 520, 'c')], '#6a4c7c', smooth=False,
           feather=0.8)
    f.fill([(444, 262, 'c'), (496, 262, 'c'), (496, 276, 'c'), (444, 276, 'c')], '#f6c8e6', smooth=False, feather=1.5)
    f.fill(ellipse(470, 222, 3, 3, 8), '#ff4d6a', feather=1.2)
    # highway ribbon of lights
    for i in range(80):
        xx = 250 + i * 5
        yy = 470 + 12 * np.sin(i / 11)
        f.fill(ellipse(xx, yy, 1.6, 1.0, 6), '#ffe6b0' if i % 3 else '#ff6a5a')


def tower(f, rs, x0, x1, top, base, cell=(30, 46), lit_p=0.55):
    """A near tower: teal facade, mullions, lit windows with interiors."""
    f.fill([(x0, base, 'c'), (x0, top, 'c'), (x1, top, 'c'), (x1, base, 'c')], '#122636', smooth=False,
           grad=(0, top, 0, base, '#16304a', '#0b1520'))
    cw, ch = cell
    cols = int((x1 - x0 - 10) / cw)
    rows = int((base - top - 10) / ch)
    for r in range(rows):
        for c in range(cols):
            wx = x0 + 10 + c * cw + rs.normal(0, 0.7)       # painted by hand, never on a grid
            wy = top + 12 + r * ch + rs.normal(0, 0.6)
            ww, wh = cw * rs.uniform(0.56, 0.66), ch * rs.uniform(0.56, 0.66)
            rect = [(wx, wy, 'c'), (wx + ww, wy, 'c'), (wx + ww, wy + wh, 'c'), (wx, wy + wh, 'c')]
            if rs.random() < lit_p:
                base_c = '#%02x%02x%02x' % tuple(int(v) for v in np.clip(
                    np.array([222, 222, 150]) + rs.normal(0, 1, 3) * [14, 14, 22] - rs.uniform(0, 40) * np.array([1, 1, .6]), 0, 255))
                f.fill(rect, base_c, smooth=False)
                # interiors: a blind, a lamp, a shape of a person or furniture
                k = rs.random()
                if k < 0.35:
                    f.fill([(wx, wy, 'c'), (wx + ww, wy, 'c'), (wx + ww, wy + wh * rs.uniform(.25, .6), 'c'),
                            (wx, wy + wh * rs.uniform(.25, .6), 'c')], '#9d9655', smooth=False)
                elif k < 0.6:   # a figure, a plant, a shelf: each one different
                    px = wx + ww * rs.uniform(.1, .7)
                    sw_, sh_ = ww * rs.uniform(.15, .45), wh * rs.uniform(.3, .75)
                    pts = [(px, wy + wh, 'c')]
                    for t in np.linspace(0, 1, 4):
                        pts.append((px + sw_ * t + rs.normal(0, 1.2), wy + wh - sh_ * rs.uniform(.6, 1)))
                    pts.append((px + sw_, wy + wh, 'c'))
                    f.fill(pts, ['#7c7a40', '#8a8448', '#6d6a3a'][rs.integers(3)])
                elif k < 0.75:
                    f.fill(ellipse(wx + ww * .5, wy + wh * .35, ww * .22, wh * .16, 10), '#fbf8d0', feather=1.2)
                f.fill([(wx + ww * .48, wy, 'c'), (wx + ww * .52, wy, 'c'), (wx + ww * .52, wy + wh, 'c'),
                        (wx + ww * .48, wy + wh, 'c')], '#4e5133', smooth=False)   # window bar
                if rs.random() < 0.04:
                    f.fill(ellipse(wx + ww / 2, wy + wh / 2, ww * .7, wh * .6, 12), '#e0e1a0', feather=2.5)
            else:
                f.fill(rect, ['#06151c', '#0a1c28'][rs.integers(2)], smooth=False)
    # Mizutani's small structure: floor ledges every few storeys, a lit edge,
    # a drainpipe painted as a flat strip with a highlight
    for r in range(0, rows, 4):
        y = top + 8 + r * ch
        f.fill([(x0, y, 'c'), (x1, y, 'c'), (x1, y + 5, 'c'), (x0, y + 5, 'c')], '#1f3a4d', smooth=False)
    f.fill([(x1 - 7, top, 'c'), (x1 - 3, top, 'c'), (x1 - 3, base, 'c'), (x1 - 7, base, 'c')], '#2e4b5c', smooth=False)
    px = x0 + (x1 - x0) * 0.62
    f.fill([(px, top + 30, 'c'), (px + 7, top + 30, 'c'), (px + 7, base, 'c'), (px, base, 'c')], '#0a1720', smooth=False)
    f.fill([(px + 1, top + 30, 'c'), (px + 2.5, top + 30, 'c'), (px + 2.5, base, 'c'), (px + 1, base, 'c')], '#2b4656', smooth=False)


def parapet(f, rs):
    wall = [(-10, 718, 'c'), (-10, 560, 'c'), (1290, 548, 'c'), (1290, 718, 'c')]
    w = f.fill(wall, '#5a2943', smooth=False, grad=(0, 560, 0, 718, '#62304c', '#3e1c33'))
    f.fill([(-10, 548, 'c'), (1290, 536, 'c'), (1290, 552, 'c'), (-10, 564, 'c')], '#7a5a86', smooth=False)
    f.fill([(-10, 564, 'c'), (1290, 552, 'c'), (1290, 562, 'c'), (-10, 574, 'c')], '#2c1e28', smooth=False)
    f.ink([(-10, 548), (1290, 536)], w=2.6, color=INK, taper=0, smooth=False)
    f.ink([(-10, 574), (1290, 562)], w=2.0, color=INK, taper=0, smooth=False)
    # grime: spatter specks and thin rain drips, painted hard like the film
    for _ in range(900):
        x, y = rs.uniform(0, 1280), rs.uniform(570, 718)
        r = rs.uniform(0.6, 2.4) if rs.random() < 0.9 else rs.uniform(3, 7)
        f.fill(ellipse(x, y, r, r * rs.uniform(.5, 1), 7, rs.uniform(0, 3)),
               ['#2c1e28', '#3a1f30', '#24141f'][rs.integers(3)], smooth=True)
    for _ in range(40):
        x, y = rs.uniform(0, 1280), rs.uniform(574, 600)
        L = rs.uniform(20, 90)
        f.fill([(x, y), (x + 3, y), (x + 2.2, y + L), (x + 1.2, y + L + 3)], '#33192a', feather=0.6)
    for _ in range(7):  # cracks
        x, y = rs.uniform(0, 1280), rs.uniform(580, 700)
        pts = [(x, y)]
        for _ in range(rs.integers(3, 6)):
            x += rs.uniform(-20, 20)
            y += rs.uniform(8, 22)
            pts.append((x, y))
        f.ink(pts, w=1.6, color=INK, taper=0.4)


def rider(f):
    """Seated on the parapet, back to camera, looking out over the city."""
    RED, RED_SH, RIM = '#8a1622', '#4c0916', '#d76a9a'
    HAIR = '#140d14'
    # torso: broad shoulders hunched forward, the jacket bunched at the waist
    back = [(904, 544, 'c'), (898, 500), (892, 458), (900, 432), (930, 416), (962, 412), (996, 416),
            (1026, 430), (1036, 456), (1030, 500), (1026, 544, 'c')]
    p = f.fill(back, RED)
    f.fill([(966, 404), (1040, 420), (1044, 560), (978, 560), (986, 480)], RED_SH, inside=p)
    f.fill([(898, 500), (930, 520), (968, 524), (1004, 518), (1032, 500), (1032, 548), (896, 548)], RED_SH,
           inside=p)
    f.ink(back, w=2.6, color=INK)
    for fold in ([(926, 470), (944, 500)], [(1004, 466), (990, 498)], [(920, 520), (960, 512), (1000, 518)],
                 [(964, 430), (966, 490)]):
        f.ink(fold, w=1.5, color=INK, taper=0.45)
    f.ink([(896, 476), (900, 440), (928, 420), (950, 414)], w=2.4, color=RIM, taper=0.3)
    # upper arms reaching forward to the knees, past the body
    for arm in ([(900, 440), (884, 470), (886, 506), (900, 512), (906, 470)],
                [(1030, 440), (1046, 472), (1044, 506), (1028, 512), (1024, 470)]):
        f.fill(arm, RED_SH)
        f.ink(arm, w=2.2, color=INK, closed=True)
    # neck and head, hair blown back
    f.fill([(948, 412), (976, 410), (978, 396), (950, 398)], '#6f4b4a')
    head = [(938, 398, 'c'), (930, 372), (934, 346), (950, 330), (972, 328), (990, 340), (996, 364),
            (990, 392), (980, 404, 'c')]
    f.fill(head, HAIR)
    f.fill([(944, 340), (926, 326, 'c'), (952, 332), (948, 314, 'c'), (970, 326), (980, 312, 'c'),
            (988, 332), (1006, 326, 'c'), (996, 346)], HAIR)
    f.ink([(944, 340), (926, 326), (952, 332), (948, 314), (970, 326), (980, 312), (988, 332), (1006, 326),
           (996, 350), (996, 364), (990, 392), (980, 404)], w=2.2, color=INK, taper=0.1, smooth=False)
    f.ink([(938, 398), (930, 372), (934, 346)], w=2.2, color=INK, taper=0.2)
    f.ink([(932, 380), (934, 352), (944, 336)], w=2.0, color=RIM, taper=0.4)
    # the sitting line where the jacket meets the ledge
    f.fill([(904, 542, 'c'), (1030, 540, 'c'), (1034, 556, 'c'), (900, 558, 'c')], '#2b0a14', smooth=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(12)
    f = Frame()
    sky(f)
    far_city(f, rs)
    tower(f, rs, -20, 236, -20, 560, cell=(28, 40), lit_p=0.5)       # left tower, cut by frame
    tower(f, rs, 640, 760, 90, 560, cell=(16, 24), lit_p=0.35)       # slim mid tower
    tower(f, rs, 790, 1300, -20, 560, cell=(26, 38), lit_p=0.6)      # big right tower
    parapet(f, rs)
    rider(f)
    (OUT / 'o02_rooftop.json').write_text(json.dumps(f.scene(), separators=(',', ':')), encoding='utf8')
    f.shoot(OUT / 'o02_rooftop.png', lens=0.6, grain=1.45, seed=2)
    print('wrote', OUT / 'o02_rooftop.png')


if __name__ == '__main__':
    main()
