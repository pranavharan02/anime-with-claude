"""Plate 04: the white dome. Effects plate.

A hard-edged sphere of light swells out of the city. Everything else drops to
silhouette, rim-lit on the edges facing the light. Light rays leak through the
gaps between towers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import skia

from engine.core import W, H, surface, to_np, flatten, save, col, hx, mix
from engine.city import rect
from engine.fx import bloom, lin_grad, radial_grad, blur
from engine.ink import path_of
from engine.paint import gouache
from engine.film import develop

OUT = ROOT / 'out' / 'akira'
CX, CY, R = 960.0, 790.0, 450.0


def zoom_rays(mask, cx, cy, steps=28, reach=0.45):
    """Radial (zoom) blur from a centre: crepuscular rays."""
    h, w = mask.shape[:2]
    acc = np.zeros_like(mask)
    for i in range(steps):
        s = 1 + reach * i / steps
        m = np.float32([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]])
        acc += cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_LINEAR) * (1 - i / steps)
    return acc / steps * 2.2


def skyline(c, rs, base, hmin, hmax, wmin, wmax, color, rim_a, rim_w):
    x = -60
    while x < W + 60:
        w = rs.uniform(wmin, wmax)
        h = rs.uniform(hmin, hmax) * (1 + 1.4 * (abs(x - CX) / W) ** 1.5)
        top = base - h
        c.drawRect(skia.Rect(x, top, x + w, H + 10), skia.Paint(AntiAlias=True, Color4f=col(color)))
        if rs.random() < 0.35:  # setback crown
            c.drawRect(skia.Rect(x + w * .18, top - h * .12, x + w * .78, top + 1),
                       skia.Paint(AntiAlias=True, Color4f=col(color)))
            top -= h * .12
        if rs.random() < 0.25:  # mast
            c.drawRect(skia.Rect(x + w * .5 - 1.5, top - h * .25, x + w * .5 + 1.5, top + 1),
                       skia.Paint(AntiAlias=True, Color4f=col(color)))
        # rim light on the edge that faces the dome, stronger nearer to it
        mid = x + w / 2
        near = np.exp(-abs(mid - CX) / 700)
        ex = x + w if mid < CX else x
        p = skia.Paint(AntiAlias=True, Color4f=col('#eaf6ff', rim_a * near),
                       StrokeWidth=rim_w, Style=skia.Paint.kStroke_Style)
        c.drawLine(ex, top, ex, base + 200, p)
        c.drawLine(x, top, x + w, top, skia.Paint(AntiAlias=True, Color4f=col('#cfe6ff', rim_a * near * 0.6),
                                                  StrokeWidth=rim_w * 0.7))
        x += w * rs.uniform(0.6, 1.0)


def render(seed=4):
    rs = np.random.default_rng(seed)
    ps, es, occ_s = surface(), surface(), surface()
    c, e, occ = ps.getCanvas(), es.getCanvas(), occ_s.getCanvas()

    # sky: night pushed violet, brightening toward the light
    lin_grad(c, path_of(rect(0, 0, W, H)), (0, 0), (0, CY), ['#07081c', '#1c1d4a', '#5a5ea6'], [0, 0.55, 1])
    glow = path_of(rect(0, 0, W, H))
    radial_grad(c, glow, (CX, CY), 1300, [('#c9d8ff', 0.55), ('#8e9be0', 0.25), ('#3a3f80', 0.0)], [0, 0.35, 1])

    # the dome: near-white, faint cool falloff to the rim, a hard edge
    dome = skia.Path()
    dome.addCircle(CX, CY, R)
    radial_grad(c, dome, (CX, CY - R * 0.15), R * 1.05,
                [('#ffffff', 1), ('#f3f8ff', 1), ('#d4e6ff', 1)], [0, 0.7, 1])
    e.drawPath(dome, skia.Paint(AntiAlias=True, Color4f=col('#e8f0ff', 0.9)))
    # shock rings racing outward over the city
    for rx, ry, a, wdt in ((R * 1.75, R * 0.2, 0.8, 4), (R * 2.5, R * 0.3, 0.45, 3), (R * 3.4, R * 0.42, 0.25, 2)):
        ring = skia.Path()
        ring.addOval(skia.Rect(CX - rx, CY - ry - 10, CX + rx, CY + ry - 10))
        for cv, aa in ((c, a), (e, a * 0.7)):
            cv.drawPath(ring, skia.Paint(AntiAlias=True, Color4f=col('#f2f7ff', aa),
                                         Style=skia.Paint.kStroke_Style, StrokeWidth=wdt))

    # silhouettes, far to near (also drawn black into the occluder for rays)
    for cv in (c, occ):
        is_occ = cv is occ
        r2 = np.random.default_rng(seed)
        skyline(cv, r2, 820, 40, 150, 25, 70, '#000000' if is_occ else '#1d2048', 0 if is_occ else 0.5, 1.4)
        skyline(cv, r2, 900, 80, 240, 40, 110, '#000000' if is_occ else '#10122e', 0 if is_occ else 0.75, 2.0)
        skyline(cv, r2, 1010, 110, 300, 60, 170, '#000000' if is_occ else '#07081a', 0 if is_occ else 0.9, 2.6)
    # a lone tower in the foreground, left, catching hard light down one edge
    tw = [(80, 70), (250, 40), (262, 1100), (60, 1100)]
    c.drawPath(path_of(np.array(tw)), skia.Paint(AntiAlias=True, Color4f=col('#05060f')))
    occ.drawPath(path_of(np.array(tw)), skia.Paint(AntiAlias=True, Color4f=col('#000000')))
    c.drawLine(250, 40, 262, 1100, skia.Paint(AntiAlias=True, Color4f=col('#f0f7ff', 0.95), StrokeWidth=4))
    e.drawLine(250, 40, 262, 1100, skia.Paint(AntiAlias=True, Color4f=col('#f0f7ff', 0.4), StrokeWidth=4))

    paint = flatten(to_np(ps))
    paint = gouache(paint, seed, amt=0.035, streak='h')
    emit = to_np(es)[..., :3]
    # rays: dome light minus everything in front of it, smeared outward
    occl = 1 - to_np(occ_s)[..., 3:4]
    rays = zoom_rays(emit * occl, CX, CY) * hx('#cfdcff')
    vis = emit * occl
    rays *= 0.2 + 0.8 * occl          # rays live in the air between towers
    img = paint + 0.3 * rays + bloom(vis, ((3, .12), (24, .18), (70, .3))) + bloom(emit, ((160, .12),))
    return img


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    img = render()
    save(develop(img, frame=7, seed=4, lift='#0b0c22'), OUT / 'p04_dome.png')
    print('wrote', OUT / 'p04_dome.png')
