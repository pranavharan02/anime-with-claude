"""Plate 01: the night city. Background painting test.

Elevated view across a megacity at night: layered towers in haze, a central
mega-tower, an elevated expressway with traffic streams, searchlights.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import skia

from engine.core import W, H, surface, to_np, flatten, save, col, hx, mix
from engine.city import City, NIGHT, rect
from engine.fx import bloom, lin_grad, ribbon
from engine.ink import path_of, curve
from engine.paint import gouache
from engine.film import develop

OUT = ROOT / 'out' / 'akira'


def sky(c, horizon):
    lin_grad(c, path_of(rect(0, 0, W, H)), (0, 0), (0, horizon + 80),
             [NIGHT['sky_top'], NIGHT['sky_mid'], NIGHT['sky_low']], [0, 0.55, 1])


def searchlights(c, e, rs, n=4):
    for i in range(n):
        x0 = rs.uniform(250, W - 250)
        y0 = rs.uniform(600, 700)
        ang = np.radians(rs.uniform(-28, 28) - 90)
        L = 1300
        spread = np.radians(rs.uniform(1.6, 2.6))
        tip_l = (x0 + L * np.cos(ang - spread), y0 + L * np.sin(ang - spread))
        tip_r = (x0 + L * np.cos(ang + spread), y0 + L * np.sin(ang + spread))
        poly = np.array([(x0, y0), tip_l, tip_r])
        end = (x0 + L * np.cos(ang), y0 + L * np.sin(ang))
        for canvas, a in ((c, 0.13), (e, 0.05)):
            cs = [col('#bfe9ff', a).toColor(), col('#bfe9ff', 0).toColor()]
            sh = skia.GradientShader.MakeLinear([skia.Point(x0, y0), skia.Point(*end)], cs)
            canvas.drawPath(path_of(poly), skia.Paint(AntiAlias=True, Shader=sh))


def expressway(c, e, rs, y, amp):
    """An elevated road sweeping across the middle ground, with traffic."""
    xs = np.linspace(-100, W + 100, 60)
    ys = y + amp * np.sin((xs / W) * np.pi * 1.2 + 0.4)
    centre = np.stack([xs, ys], 1)
    deck = ribbon(centre, np.full(len(xs), 34))
    c.drawPath(path_of(deck), skia.Paint(AntiAlias=True, Color4f=col('#0b1a24')))
    # deck side face and parapet light line
    side = ribbon(centre + [0, 22], np.full(len(xs), 14))
    c.drawPath(path_of(side), skia.Paint(AntiAlias=True, Color4f=col('#132b37')))
    par = skia.Paint(AntiAlias=True, Color4f=col('#6fc8c4', 0.7), Style=skia.Paint.kStroke_Style, StrokeWidth=1.6)
    c.drawPath(path_of(centre + [0, -16], closed=False), par)
    # piers
    for x in range(0, W, 170):
        yy = y + amp * np.sin((x / W) * np.pi * 1.2 + 0.4)
        pier = rect(x - 9, yy + 28, 18, H)
        lin_grad(c, path_of(pier), (0, yy), (0, yy + 260), ['#0d1e29', '#2c6668'])
    # traffic: two lanes of light streaks, white heads one way, red tails the other
    dense = curve([tuple(p) for p in centre[::3]], step=2.0)
    for lane, colr, off in ((0, '#fff4d6', -6), (1, '#ff3b2a', 7)):
        k = 0
        while k < len(dense) - 30:
            run = rs.integers(8, 40)
            seg = dense[k:k + run] + [0, off]
            for canvas, w, a in ((c, 2.4, 0.95), (e, 3.0, 0.9)):
                p = skia.Paint(AntiAlias=True, Color4f=col(colr, a), Style=skia.Paint.kStroke_Style,
                               StrokeWidth=w, StrokeCap=skia.Paint.kRound_Cap)
                canvas.drawPath(path_of(seg, closed=False), p)
            k += run + rs.integers(10, 60)


def megatower(city, x, base, w, h):
    """The landmark: a tall stepped tower with a lit crown and beacon."""
    city.tower(x, base, w, h * 0.62, 0.5, kind='box')
    city.tower(x + w * 0.12, base - h * 0.62, w * 0.76, h * 0.24, 0.5, kind='box')
    city.tower(x + w * 0.24, base - h * 0.86, w * 0.52, h * 0.1, 0.5, kind='cyl')
    # crown ring
    cy = base - h * 0.96
    for canvas, a in ((city.c, 0.9), (city.e, 0.6)):
        p = skia.Paint(AntiAlias=True, Color4f=col('#ffcf7a', a), Style=skia.Paint.kStroke_Style, StrokeWidth=3)
        canvas.drawOval(skia.Rect(x + w * 0.2, cy - 6, x + w * 0.8, cy + 6), p)
    mast = skia.Paint(AntiAlias=True, Color4f=col('#1c3a47'), StrokeWidth=4)
    city.c.drawLine(x + w / 2, cy, x + w / 2, cy - h * 0.18, mast)
    city._light(x + w / 2, cy - h * 0.18, 4, '#ff4433', 1.0)


def render(seed=11):
    rs = np.random.default_rng(seed)
    ps, es = surface(), surface()
    c, e = ps.getCanvas(), es.getCanvas()
    city = City(c, e, seed=seed)

    sky(c, 620)
    searchlights(c, e, rs)

    city.row(-80, W + 80, 640, 60, 200, 0.97, density=1.4)
    city.haze(520, 650, a1=0.55)
    city.row(-80, W + 80, 690, 110, 330, 0.82, density=1.2)
    city.haze(560, 700, a1=0.6)
    megatower(city, 880, 760, 150, 700)
    city.row(-60, 820, 760, 150, 430, 0.62)
    city.row(1080, W + 60, 760, 150, 430, 0.62)
    city.haze(640, 770, a1=0.55)
    city.row(-60, W + 60, 900, 180, 520, 0.4, density=0.9)
    city.haze(760, 905, a1=0.5)
    expressway(c, e, rs, 860, 50)
    city.haze(930, 1040, a1=0.35)
    # near framing towers
    city.tower(-40, H + 10, 300, H + 40, 0.08, side=0.18, kind='box')
    city.tower(1700, H + 10, 270, H - 120, 0.1, side=-0.2, kind='stepped')

    paint = flatten(to_np(ps))
    paint = gouache(paint, seed, amt=0.05)
    emit = to_np(es)[..., :3]
    img = paint + bloom(emit)
    return img


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    img = render()
    save(img, OUT / 'p01_city_raw.png')
    save(develop(img, frame=0, seed=1), OUT / 'p01_city.png')
    print('wrote', OUT / 'p01_city.png')
