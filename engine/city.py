"""Night-city background painter in the manner of a poster-colour BG.

Draws frontal tower elevations in depth layers. Lit windows also go to an
emission canvas so the film pass can bloom them.
"""
import numpy as np
import skia

from .core import col, mix, hx
from .fx import lin_grad
from .ink import path_of

NIGHT = {
    'sky_top': '#040817', 'sky_mid': '#0b2133', 'sky_low': '#1d5560',
    'near': '#0b1c27', 'far': '#5d9aa0', 'haze': '#5aa6a0',
    'edge': '#7fd2cc', 'win_dark': '#071620',
    'lights': ['#fff3c9', '#ffe39a', '#fff8e6', '#ffd27a', '#a8f1ff', '#ff9c7a'],
}


def rect(x, y, w, h):
    return np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], float)


class City:
    def __init__(self, paint_canvas, emit_canvas, seed=0, pal=NIGHT):
        self.c = paint_canvas
        self.e = emit_canvas
        self.rs = np.random.default_rng(seed)
        self.p = pal

    # -- one tower ---------------------------------------------------------
    def tower(self, x, base, w, h, depth, side=0.0, kind='box', detail=1.0):
        """depth 0 = nearest, 1 = farthest. side>0 shows a right face, <0 left."""
        rs, p = self.rs, self.p
        face = mix(p['near'], p['far'], depth)
        face = face * rs.uniform(0.85, 1.12) + rs.normal(0, 0.012, 3)
        glow = mix(face, p['haze'], 0.55 + 0.25 * depth)
        sidec = face * np.array([0.62, 0.7, 0.82])
        top = base - h
        sw = abs(side) * w
        self.style = rs.choice(['grid', 'grid', 'sparse', 'sparse', 'bands', 'fins'])

        # side face (drawn first, a narrow plane receding to the vanishing point)
        if sw > 0.5:
            sx = x + w if side > 0 else x - sw
            lift = sw * 0.18
            if side > 0:
                poly = np.array([(x + w, top), (x + w + sw, top + lift), (x + w + sw, base), (x + w, base)])
            else:
                poly = np.array([(x - sw, top + lift), (x, top), (x, base), (x - sw, base)])
            lin_grad(self.c, path_of(poly), (0, top), (0, base),
                     [sidec, mix(sidec, p['haze'], 0.45 + 0.3 * depth)])
            self._windows(poly, sx, top + lift, sw, h - lift, depth, dim=0.55, cols_scale=0.5)

        # front face, possibly with setbacks / special crowns
        if kind == 'stepped':
            # stacked boxes, each setback narrower and higher
            hs = rs.uniform(0.15, 0.45, rs.integers(1, 3))
            bounds = sorted({base, top, *(base - h * (1 - f) for f in hs)}, reverse=True)
            for i in range(len(bounds) - 1):
                wi = w * (1 - 0.16 * i)
                xi = x + (w - wi) / 2
                self._face(xi, bounds[i + 1], wi, bounds[i] - bounds[i + 1], face, glow, depth, detail, base)
        elif kind == 'cyl':
            self._cylinder(x, top, w, h, face, glow, depth, base)
        else:
            self._face(x, top, w, h, face, glow, depth, detail, base)

        # crowns and antennas
        if kind == 'antenna' or rs.random() < 0.18 * (1 - depth):
            ax = x + w * rs.uniform(0.3, 0.7)
            ah = h * rs.uniform(0.08, 0.22)
            paint = skia.Paint(AntiAlias=True, Color4f=col(face * 0.8), StrokeWidth=max(1, 3 * (1 - depth)),
                               Style=skia.Paint.kStroke_Style)
            self.c.drawLine(ax, top, ax, top - ah, paint)
            self._light(ax, top - ah, 2.2 * (1 - depth) + 1, '#ff5040', 1.0)
        if kind == 'slant':
            poly = np.array([(x, top), (x + w, top - w * 0.35), (x + w, top + 1), (x, top + 1)])
            self.c.drawPath(path_of(poly), skia.Paint(AntiAlias=True, Color4f=col(face * 1.05)))

    def _face(self, x, top, w, h, face, glow, depth, detail, base):
        poly = rect(x, top, w, h)
        lin_grad(self.c, path_of(poly), (0, top), (0, base), [face, face, glow], [0, 0.35, 1])
        # painted lit edge on the corner nearest the light
        ec = mix(face, self.p['edge'], 0.55 - 0.3 * depth)
        ep = skia.Paint(AntiAlias=True, Color4f=col(ec), StrokeWidth=max(0.8, 2.2 * (1 - depth)),
                        Style=skia.Paint.kStroke_Style)
        self.c.drawLine(x + 0.5, top, x + 0.5, base, ep)
        self.c.drawLine(x, top + 0.5, x + w, top + 0.5, ep)
        self._windows(poly, x, top, w, h, depth, detail=detail)

    def _cylinder(self, x, top, w, h, face, glow, depth, base):
        poly = rect(x, top, w, h)
        cols = [face * 0.7, mix(face, self.p['edge'], 0.35), face, face * 0.6]
        lin_grad(self.c, path_of(poly), (x, 0), (x + w, 0), cols, [0, 0.3, 0.6, 1])
        # dome cap
        cap = skia.Path()
        cap.addOval(skia.Rect(x, top - w * 0.12, x + w, top + w * 0.12))
        self.c.drawPath(cap, skia.Paint(AntiAlias=True, Color4f=col(mix(face, self.p['edge'], 0.3))))
        # horizontal window bands
        rs = self.rs
        band = max(3, 14 * (1 - depth))
        y = top + band
        while y < base - 2:
            if rs.random() < 0.55:
                lit = rs.choice(self.p['lights'][:4])
                a = rs.uniform(0.35, 0.9)
                bp = rect(x + w * 0.08, y, w * 0.84, max(1, band * 0.28))
                self.c.drawPath(path_of(bp), skia.Paint(AntiAlias=True, Color4f=col(lit, a * 0.8)))
                self.e.drawPath(path_of(bp), skia.Paint(AntiAlias=True, Color4f=col(lit, a * 0.25)))
            y += band

    def _windows(self, clip_poly, x, top, w, h, depth, dim=1.0, cols_scale=1.0, detail=1.0):
        rs, p = self.rs, self.p
        style = getattr(self, 'style', 'grid')
        near = (1 - depth)
        cw = (2.4 + 4.6 * near ** 1.6) * rs.uniform(0.8, 1.2) * cols_scale ** 0.3
        chh = cw * rs.uniform(1.3, 1.8)
        gap = rs.uniform(0.35, 0.55)
        ncol = max(1, int(w / cw))
        nrow = max(1, int(h / chh))
        if ncol * nrow > 60000:
            return
        # clustered occupancy: some floors and bays are lit together
        density = {'grid': 1.0, 'sparse': 0.35, 'bands': 1.0, 'fins': 1.0}[style]
        floor_on = rs.random(nrow) < rs.uniform(0.2, 0.6) * density
        bay_on = rs.random(ncol) < rs.uniform(0.35, 0.8)
        lit = (rs.random((nrow, ncol)) < rs.uniform(0.03, 0.18) * density) | (
            floor_on[:, None] & bay_on[None, :] & (rs.random((nrow, ncol)) < 0.75))
        if rs.random() < 0.6:  # a dark section: offices closed for the night
            r0 = rs.integers(0, nrow)
            lit[r0:r0 + rs.integers(2, max(3, nrow // 2))] = False
        pal = p['lights']
        main = rs.choice(pal[:4])
        haze = hx(p['haze'])
        a = (0.4 + 0.6 * near) * dim
        self.c.save()
        self.c.clipPath(path_of(clip_poly), skia.ClipOp.kIntersect, True)
        dark, groups = skia.Path(), {}
        cellw, cellh = w / ncol, h / nrow
        for r in range(nrow):
            wy = top + r * cellh + cellh * gap * 0.5 + chh * 0.5
            if wy > top + h - 2:
                continue
            if style == 'bands':
                if r % 2:
                    continue
                c = 0
                while c < ncol:
                    run = int(rs.integers(1, 8))
                    rect_ = skia.Rect.MakeXYWH(x + c * cellw, wy, run * cellw, cellh * 0.55)
                    if lit[r, c]:
                        groups.setdefault(main, skia.Path()).addRect(rect_)
                    else:
                        dark.addRect(rect_)
                    c += run
                continue
            for c in range(ncol):
                if style == 'fins' and c % 3 == 2:
                    continue
                wx = x + c * cellw + cellw * gap * 0.5
                ww, wh = cellw * (1 - gap), cellh * (1 - gap)
                if style == 'fins':
                    ww, wh = cellw * 0.8, cellh * 1.02
                rect_ = skia.Rect.MakeXYWH(wx, wy, ww, wh)
                if lit[r, c]:
                    k = main if rs.random() < 0.85 else pal[rs.integers(len(pal))]
                    groups.setdefault(k, skia.Path()).addRect(rect_)
                elif depth < 0.75 and detail > 0.5:
                    dark.addRect(rect_)
        self.c.drawPath(dark, skia.Paint(AntiAlias=True, Color4f=col(p['win_dark'], 0.45 * near)))
        for k, path in groups.items():
            kc = mix(k, haze, 0.55 * depth)
            self.c.drawPath(path, skia.Paint(AntiAlias=True, Color4f=col(kc, a)))
            self.e.drawPath(path, skia.Paint(AntiAlias=True, Color4f=col(kc, a * 0.16 * near)))
        self.c.restore()

    def _light(self, x, y, r, c, a):
        p = skia.Paint(AntiAlias=True, Color4f=col(c, a))
        self.c.drawCircle(x, y, r, p)
        self.e.drawCircle(x, y, r * 1.5, skia.Paint(AntiAlias=True, Color4f=col(c, a)))

    # -- whole layers --------------------------------------------------------
    def row(self, x0, x1, base, hmin, hmax, depth, vx=None, density=1.0, kinds=None):
        """A row of towers between x0..x1 standing on y=base."""
        rs = self.rs
        vx = (x0 + x1) / 2 if vx is None else vx
        kinds = kinds or ['box'] * 6 + ['stepped'] * 2 + ['cyl', 'antenna', 'slant']
        x = x0 - rs.uniform(0, 60)
        items = []
        while x < x1:
            w = rs.uniform(40, 150) * (1.25 - 0.8 * depth) * rs.choice([1, 1, 1, 1.6])
            h = rs.uniform(hmin, hmax) * (rs.choice([1, 1, 1.3]) if rs.random() < 0.2 else 1)
            items.append((x, w, h, rs.choice(kinds)))
            x += w * rs.uniform(0.55, 1.05) / density
        # draw tallest first so shorter ones overlap them, like stacked flats
        for x, w, h, kind in sorted(items, key=lambda t: -t[2]):
            off = (x + w / 2 - vx) / (x1 - x0)
            side = np.clip(-off * 0.5, -0.25, 0.25) * (1 - depth)
            self.tower(x, base, w, h, depth, side=side, kind=kind)

    def haze(self, y_top, y_bot, color=None, a0=0.0, a1=0.75, w=None):
        w = w or self.c.getBaseLayerSize().width()
        color = color or self.p['haze']
        poly = rect(-10, y_top, w + 20, y_bot - y_top)
        c0, c1 = col(color, a0).toColor(), col(color, a1).toColor()
        shader = skia.GradientShader.MakeLinear([skia.Point(0, y_top), skia.Point(0, y_bot)], [c0, c1])
        self.c.drawPath(path_of(poly), skia.Paint(AntiAlias=True, Shader=shader))
