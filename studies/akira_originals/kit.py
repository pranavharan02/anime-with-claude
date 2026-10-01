"""Authoring kit for original frames in the scene format the recreations use.

You draw with three tools, the same three layers the traced film frames have:
  sky()/paint_field()  the airbrush layer: paint a soft field, sampled to a mesh
  fill()               a cel paint region (control points, flat or gradient)
  ink()                a traced line with a tapered width profile

Defaults come from measurements of five real frames (see recreate.py):
ink is near-black aubergine, 2-3 px inside, 4-5 px on outer contours at
1280 x 718; only a few regions have soft edges; the print is muted.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from engine.ink import curve
from engine.scene import render, _hex

W, H = 1280, 718
INK = '#1a060c'


class Frame:
    def __init__(self, w=W, h=H, mesh_step=12):
        self.w, self.h, self.step = w, h, mesh_step
        self.field = np.zeros((h, w, 3), np.float32)
        self.fills, self.lines = [], []
        self._rs = np.random.default_rng(99)

    # -- airbrush ------------------------------------------------------------
    def paint_field(self, fn):
        """fn(xx, yy) -> (h, w, 3) float RGB 0..255: the soft painted field."""
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.field = fn(xx, yy).astype(np.float32)

    def _mesh(self):
        st = self.step
        cols = int(np.ceil((self.w - 1) / st)) + 1
        rows = int(np.ceil((self.h - 1) / st)) + 1
        gx = np.minimum(np.arange(cols) * st, self.w - 1)
        gy = np.minimum(np.arange(rows) * st, self.h - 1)
        f = cv2.GaussianBlur(self.field, (0, 0), st * 0.4)
        cs = f[gy[:, None], gx[None, :]].reshape(-1, 3)
        return dict(step=st, cols=cols, rows=rows, colors=[_hex(c) for c in cs])

    # -- paint ---------------------------------------------------------------
    def fill(self, pts, color, feather=0.0, smooth=True, grad=None, inside=None):
        """grad = (x0, y0, x1, y1, color0, color1) for a linear gradient fill.
        inside = another shape (control points or polygon) to clip to, the way
        a shadow is painted only within its material."""
        poly = curve(pts, closed=True) if smooth else np.array([p[:2] for p in pts], float)
        if inside is not None:
            clip = inside if isinstance(inside, np.ndarray) else curve(inside, closed=True)
            g = Polygon(poly).buffer(0).intersection(Polygon(clip).buffer(0))
            if g.is_empty:
                return None
            if g.geom_type != 'Polygon':
                g = max(getattr(g, 'geoms', [g]), key=lambda q: q.area)
            poly = np.array(g.exterior.coords)[:-1]
        f = color if grad is None else ['lin', *grad]
        item = [f, [round(float(v), 1) for v in poly.reshape(-1)]]
        if feather:
            item.append(feather)
        self.fills.append(item)
        return poly

    # -- ink -----------------------------------------------------------------
    def ink(self, pts, w=2.4, color=INK, taper=0.25, closed=False, smooth=True, press=None):
        poly = curve(pts, closed=closed) if smooth else np.array([p[:2] for p in pts], float)
        if closed:
            poly = np.vstack([poly, poly[:1]])
        n = len(poly)
        u = np.linspace(0, 1, n)
        wid = np.full(n, w, float)
        if taper and not closed:
            ramp = np.minimum(1, np.minimum(u, 1 - u) / taper)
            wid *= 0.3 + 0.7 * np.sin(ramp * np.pi / 2)
        if press is not None:
            wid *= press(u)
        # a hand never holds a constant pressure: slow swells plus a little jitter
        ph = self._rs.uniform(0, 6.3, 3)
        L = max(1.0, float(np.hypot(*np.diff(poly, axis=0).T).sum()))
        sarc = u * L
        wid *= (1 + 0.22 * np.sin(sarc / 37 + ph[0]) + 0.12 * np.sin(sarc / 11 + ph[1])
                + 0.05 * self._rs.standard_normal(n))
        wid = np.clip(wid, 0.5, None)
        keep = np.r_[np.arange(0, n - 1, 2), n - 1] if n > 24 else np.arange(n)
        a = np.column_stack([poly[keep], wid[keep]])
        self.lines.append([color, [round(float(v), 2) for v in a.reshape(-1)]])
        return poly

    # -- output --------------------------------------------------------------
    def camera(self, scale, cx, cy, tx, ty, sy=None):
        """Re-frame the drawn cels: scale about (cx, cy) and move it to (tx, ty).
        Ink keeps its measured pixel width, like a closer setup re-inked."""
        sy = sy or scale
        X = lambda x: tx + (x - cx) * scale
        Y = lambda y: ty + (y - cy) * sy
        for it in self.fills:
            xy = it[1]
            it[1] = [round(X(v), 1) if i % 2 == 0 else round(Y(v), 1) for i, v in enumerate(xy)]
            if not isinstance(it[0], str):
                g = it[0]
                g[1], g[2], g[3], g[4] = X(g[1]), Y(g[2]), X(g[3]), Y(g[4])
        for ln in self.lines:
            a = ln[1]
            ln[1] = [round(X(v), 2) if i % 3 == 0 else round(Y(v), 2) if i % 3 == 1 else v
                     for i, v in enumerate(a)]

    def scene(self):
        return dict(w=self.w, h=self.h, mesh=self._mesh(), fills=self.fills, lines=self.lines)

    def shoot(self, path, lens=0.7, grain=1.2, seed=0, scale=1.0):
        """Render the scene code and photograph it like the print: lens + grain."""
        img = render(self.scene(), scale=scale)
        if lens:
            img = cv2.GaussianBlur(img, (0, 0), lens * scale)
        rs = np.random.default_rng(seed)
        img = print_look(img, rs)
        n = rs.standard_normal(img.shape[:2] + (1,)).astype(np.float32)
        n = n + 0.35 * rs.standard_normal(img.shape).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 0.6)
        n /= n.std() + 1e-6
        img = np.clip(img + n * grain / 255.0, 0, 1)
        Image.fromarray((img * 255 + .5).astype(np.uint8)).save(path)
        return img


def ellipse(cx, cy, rx, ry, n=16, rot=0.0):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = rx * np.cos(a), ry * np.sin(a)
    c, s = np.cos(rot), np.sin(rot)
    return [(cx + x[i] * c - y[i] * s, cy + x[i] * s + y[i] * c) for i in range(n)]


def _field(rs, h, w, sig):
    n = rs.standard_normal((h, w)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sig)
    return n / (n.std() + 1e-6)


def print_look(img, rs):
    """What the paint and the 1988 print add on top of the cel shapes."""
    h, w = img.shape[:2]
    # poster paint is never perfectly even: broad mottling + fine tooth
    m = 1 + 0.008 * _field(rs, h, w, 9) + 0.006 * _field(rs, h, w, 1.6)
    img = img * m[..., None]
    # halation: bright lights bleed warm into the emulsion
    L = img @ np.array([.299, .587, .114], np.float32)
    hi = np.clip(L - 0.62, 0, None)[..., None] * img
    img = img + 0.5 * cv2.GaussianBlur(hi, (0, 0), 7) * np.array([1.0, 0.75, 0.6], np.float32)
    # print grade: blacks lift toward warm aubergine, slight magenta in the mids
    img = 0.035 * np.array([0.16, 0.06, 0.12], np.float32) / 0.16 * 0.16 + img * 0.97
    img[..., 0] *= 1.01
    img[..., 2] *= 1.005
    # cel dust and the odd hair, caught under the camera
    for _ in range(rs.poisson(9)):
        x, y = int(rs.uniform(0, w)), int(rs.uniform(0, h))
        r = rs.uniform(0.6, 1.8)
        val = 0.92 if rs.random() < 0.55 else 0.05
        mask = np.zeros((h, w), np.float32)
        cv2.circle(mask, (x, y), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.5)[..., None] * rs.uniform(0.3, 0.7)
        img = img * (1 - mask) + val * mask
    return np.clip(img, 0, 1)
