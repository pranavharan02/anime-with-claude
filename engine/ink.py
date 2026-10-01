"""Hand-drawn vector primitives: splines, tapered ink strokes, cel fills.

Shapes are authored as control points. A point is (x, y) for a smooth knot or
(x, y, 'c') for a corner. Curves pass through every knot (centripetal
Catmull-Rom), so drawing in code feels like placing pencil marks.
"""
from contextlib import contextmanager

import cv2
import numpy as np
import skia

from .core import col

INK = '#1d1418'  # traced line colour: brown-black, never pure black


def _ccr(p0, p1, p2, p3, n):
    def tj(ti, a, b):
        return ti + max(float(np.linalg.norm(b - a)), 1e-4) ** 0.5
    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False)[:, None]
    a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
    b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
    return (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2


def curve(pts, closed=False, step=2.5):
    """Control points -> dense polyline (N, 2)."""
    xy = [np.array(p[:2], np.float64) for p in pts]
    corner = [len(p) > 2 and p[2] == 'c' for p in pts]
    n = len(xy)
    if n < 2:
        return np.array(xy)
    out = []
    spans = n if closed else n - 1
    for i in range(spans):
        i1, i2 = i, (i + 1) % n
        p1, p2 = xy[i1], xy[i2]
        if corner[i1] or (not closed and i1 == 0):
            p0 = p1
        else:
            p0 = xy[(i1 - 1) % n]
        if corner[i2] or (not closed and i2 == n - 1):
            p3 = p2
        else:
            p3 = xy[(i2 + 1) % n]
        seg = max(2, int(np.linalg.norm(p2 - p1) / step))
        if np.allclose(p1, p2):
            continue
        out.append(_ccr(p0, p1, p2, p3, seg))
    if not closed:
        out.append(xy[-1][None])
    return np.concatenate(out)


def path_of(poly, closed=True):
    p = skia.Path()
    p.moveTo(float(poly[0][0]), float(poly[0][1]))
    for x, y in poly[1:]:
        p.lineTo(float(x), float(y))
    if closed:
        p.close()
    return p


def _noise1d(s, rs, octaves=((90, 1.0), (31, 0.5), (11, 0.25))):
    v = np.zeros_like(s)
    for period, amp in octaves:
        f = 2 * np.pi / (period * rs.uniform(0.7, 1.4))
        v += amp * np.sin(s * f + rs.uniform(0, 2 * np.pi))
    return v / sum(a for _, a in octaves)


class Pen:
    """Draws onto a skia canvas in the canvas's current coordinate space."""

    def __init__(self, canvas, seed=0):
        self.c = canvas
        self.rs = np.random.default_rng(seed)

    # -- fills -------------------------------------------------------------
    def fill(self, pts, color, smooth=True, alpha=1.0):
        poly = curve(pts, closed=True) if smooth else np.array([p[:2] for p in pts])
        paint = skia.Paint(AntiAlias=True, Color4f=col(color, alpha))
        self.c.drawPath(path_of(poly), paint)
        return poly

    def fill_poly(self, poly, color, alpha=1.0):
        paint = skia.Paint(AntiAlias=True, Color4f=col(color, alpha))
        self.c.drawPath(path_of(poly), paint)

    @contextmanager
    def clip(self, pts, smooth=True):
        poly = curve(pts, closed=True) if smooth else np.array([p[:2] for p in pts])
        self.c.save()
        self.c.clipPath(path_of(poly), skia.ClipOp.kIntersect, True)
        try:
            yield poly
        finally:
            self.c.restore()

    def union_outline(self, shapes, res=3.0, eps=0.6):
        """Outline of the union of several shapes (local coords) as polylines.

        Inkers trace the silhouette of a mass of hair, not every clump, so
        rasterise the union, trace its contour and hand back closed polylines.
        """
        polys = [curve(s, closed=True) if not isinstance(s, np.ndarray) else s for s in shapes]
        allp = np.vstack(polys)
        lo = allp.min(0) - 4
        size = np.ceil((allp.max(0) + 4 - lo) * res).astype(int)
        mask = np.zeros((size[1], size[0]), np.uint8)
        for p in polys:
            cv2.fillPoly(mask, [np.round((p - lo) * res).astype(np.int32)], 255, cv2.LINE_8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        out = []
        for c in cs:
            c = cv2.approxPolyDP(c, eps * res, True)[:, 0, :].astype(float)
            out.append(c / res + lo)
        return out

    # -- ink ---------------------------------------------------------------
    def line(self, pts, w=2.0, color=INK, closed=False, smooth=True,
             taper=0.18, var=0.22, wobble=0.35, alpha=1.0):
        """Tapered, slightly uneven traced line through control points."""
        poly = curve(pts, closed=closed) if smooth else np.array([p[:2] for p in pts], float)
        if closed:
            poly = np.vstack([poly, poly[:1]])
        self.stroke(poly, w, color, taper=0 if closed else taper, var=var,
                    wobble=wobble, alpha=alpha)
        return poly

    def stroke(self, poly, w, color=INK, taper=0.18, var=0.22, wobble=0.35, alpha=1.0):
        poly = np.asarray(poly, float)
        if len(poly) < 2:
            return
        d = np.diff(poly, axis=0)
        seglen = np.hypot(d[:, 0], d[:, 1])
        s = np.concatenate([[0], np.cumsum(seglen)])
        total = max(s[-1], 1e-6)
        u = s / total
        tang = np.gradient(poly, axis=0)
        tn = np.hypot(tang[:, 0], tang[:, 1])[:, None] + 1e-9
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1) / tn
        if wobble:
            poly = poly + nrm * (wobble * _noise1d(s, self.rs))[:, None]
        width = w * (1 + var * _noise1d(s, self.rs, ((60, 1), (17, .5))))
        if taper:
            ramp = np.minimum(1.0, np.minimum(u, 1 - u) / taper)
            width = width * (0.25 + 0.75 * np.sin(ramp * np.pi / 2))
        paint = skia.Paint(AntiAlias=True, Color4f=col(color, alpha),
                           Style=skia.Paint.kStroke_Style,
                           StrokeCap=skia.Paint.kRound_Cap)
        if alpha < 1:
            self.c.saveLayerAlpha(None, int(alpha * 255))
            paint.setAlphaf(1.0)
        for i in range(len(poly) - 1):
            paint.setStrokeWidth(float(0.5 * (width[i] + width[i + 1])))
            self.c.drawLine(float(poly[i, 0]), float(poly[i, 1]),
                            float(poly[i + 1, 0]), float(poly[i + 1, 1]), paint)
        if alpha < 1:
            self.c.restore()
