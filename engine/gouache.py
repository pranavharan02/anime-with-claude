"""Gouache / poster-colour texture painting for background surfaces.

Everything works on float RGB textures (th, tw, 3) in 0..1, painted flat in a
surface's own space before stage.py warps it into perspective. The rules come
from the film's backgrounds: flat colour laid with a wide brush (faint
streaks), uneven pigment, hard but slightly irregular edges, ruler-drawn
highlights, rust and grime run down from sills, dry-brush stains, every window
its own dab.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = 'C:/Windows/Fonts/YuGothB.ttc'


def hx(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def field(rs, h, w, sy, sx):
    """Smooth noise, std 1, with correlation lengths sy (rows) and sx (cols)."""
    gh, gw = max(2, int(h / max(sy, 1))), max(2, int(w / max(sx, 1)))
    n = rs.standard_normal((gh, gw)).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-6)


def wall(h, w, color, rs, streak='v', var=0.035, dark_bottom=0.12, light_top=0.0):
    """Base coat: flat colour, brush streaks, uneven pigment, a grime falloff."""
    c = hx(color) if isinstance(color, str) else np.asarray(color, np.float32)
    if streak == 'v':
        s = field(rs, h, w, h / 2, 10)
    else:
        s = field(rs, h, w, 10, w / 2)
    m = 1 + var * (0.6 * s + 0.5 * field(rs, h, w, 60, 60) + 0.06 * field(rs, h, w, 4, 4))
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    m *= (1 - dark_bottom * y ** 2) * (1 + light_top * (1 - y) ** 2)
    return np.clip(c * m[..., None], 0, 1)


def blend(tex, mask, color, alpha=1.0):
    c = hx(color) if isinstance(color, str) else np.asarray(color, np.float32)
    a = np.clip(mask * alpha, 0, 1)[..., None]
    tex[:] = tex * (1 - a) + c * a
    return tex


def rough_mask(h, w, polys, rs, rough=0.6):
    """Filled polygons with slightly irregular, hand-cut edges."""
    m = np.zeros((h, w), np.float32)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p) * 4).astype(np.int32)], 1.0, cv2.LINE_AA, shift=2)
    if rough:
        m = cv2.GaussianBlur(m, (0, 0), 0.7)
        m = np.clip((m - 0.5) * 2.2 + 0.5 + 0.18 * rough * field(rs, h, w, 3, 3), 0, 1)
    return m


def poly(tex, rs, pts, color, alpha=1.0, rough=0.6, pad=4):
    """Fill a polygon with irregular edges, touching only its bounding box.
    Returns (mask, (y0, y1, x0, x1)) so callers can reuse the cut."""
    h, w = tex.shape[:2]
    pts = np.asarray(pts, np.float32)
    bx0, by0 = int(max(0, np.floor(pts[:, 0].min()) - pad)), int(max(0, np.floor(pts[:, 1].min()) - pad))
    bx1, by1 = int(min(w, np.ceil(pts[:, 0].max()) + pad)), int(min(h, np.ceil(pts[:, 1].max()) + pad))
    if bx1 <= bx0 or by1 <= by0:
        return None, None
    m = rough_mask(by1 - by0, bx1 - bx0, [pts - [bx0, by0]], rs, rough)
    blend(tex[by0:by1, bx0:bx1], m, color, alpha)
    return m, (by0, by1, bx0, bx1)


def dab(tex, rs, x0, y0, x1, y1, color, alpha=1.0, var=0.04, rough=0.6):
    """A rectangle laid in with one brush load: uneven fill, irregular edge."""
    c = hx(color) if isinstance(color, str) else np.asarray(color, np.float32)
    shade = 1 + var * rs.standard_normal()
    return poly(tex, rs, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], np.clip(c * shade, 0, 1), alpha, rough)


def ruler(tex, p0, p1, color, width=1.0, alpha=1.0):
    """A ruling-pen line: straight, crisp, slightly uneven."""
    h, w = tex.shape[:2]
    m = np.zeros((h, w), np.float32)
    cv2.line(m, (int(p0[0] * 16), int(p0[1] * 16)), (int(p1[0] * 16), int(p1[1] * 16)), 1.0,
             max(1, int(round(width))), cv2.LINE_AA, shift=4)
    blend(tex, m, color, alpha)


def drips(tex, rs, n, color, x0=0, x1=None, y_from=None, length=(20, 120), width=(1, 4), alpha=(0.25, 0.7)):
    """Rust and grime running down from sills and ledges."""
    h, w = tex.shape[:2]
    x1 = w if x1 is None else x1
    m = np.zeros((h, w), np.float32)
    for _ in range(n):
        x = rs.uniform(x0, x1)
        y = rs.choice(y_from) if y_from is not None else rs.uniform(0, h * 0.7)
        L = rs.uniform(*length)
        wd = rs.uniform(*width)
        a = rs.uniform(*alpha)
        pts = []
        for t in np.linspace(0, 1, 12):
            pts.append((x + rs.normal(0, 0.4) + 1.5 * np.sin(t * 3 + x), y + L * t))
        for i in range(len(pts) - 1):
            t = i / (len(pts) - 1)
            cv2.line(m, (int(pts[i][0] * 4), int(pts[i][1] * 4)), (int(pts[i + 1][0] * 4), int(pts[i + 1][1] * 4)),
                     a * (1 - t) ** 0.7, max(1, int(wd * (1 - 0.6 * t))), cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(m, (0, 0), 0.6)
    blend(tex, m, color, 1.0)


def stains(tex, rs, n, color, size=(20, 80), alpha=(0.15, 0.4), dry=0.5):
    """Dry-brush blotches: broken, granular edges."""
    h, w = tex.shape[:2]
    grain = field(rs, h, w, 2, 2)
    m = np.zeros((h, w), np.float32)
    for _ in range(n):
        cx, cy = rs.uniform(0, w), rs.uniform(0, h)
        r = rs.uniform(*size)
        cv2.ellipse(m, (int(cx), int(cy)), (int(r), int(r * rs.uniform(0.3, 1))), rs.uniform(0, 180), 0, 360,
                    rs.uniform(*alpha), -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 6)
    m = np.clip(m * (1 + dry * grain) - dry * 0.08, 0, 1)
    blend(tex, m, color, 1.0)


def cracks(tex, rs, n, color, length=(30, 120), width=1.2, alpha=0.7):
    h, w = tex.shape[:2]
    m = np.zeros((h, w), np.float32)
    for _ in range(n):
        x, y = rs.uniform(0, w), rs.uniform(0, h)
        ang = rs.uniform(0, 2 * np.pi)
        L = rs.uniform(*length)
        steps = int(L / 6) + 2
        for _ in range(steps):
            ang += rs.normal(0, 0.45)
            nx, ny = x + 6 * np.cos(ang), y + 6 * np.sin(ang)
            cv2.line(m, (int(x * 4), int(y * 4)), (int(nx * 4), int(ny * 4)), 1.0, max(1, int(width)), cv2.LINE_AA, shift=2)
            if rs.random() < 0.12:  # a branch
                bx, by = nx + 10 * np.cos(ang + 1), ny + 10 * np.sin(ang + 1)
                cv2.line(m, (int(nx * 4), int(ny * 4)), (int(bx * 4), int(by * 4)), 0.7, 1, cv2.LINE_AA, shift=2)
            x, y = nx, ny
    blend(tex, m, color, alpha)


def shutter(tex, rs, x0, y0, x1, y1, color, pitch=7):
    """A roller shutter: corrugations as alternating ruled light and dark."""
    c = hx(color)
    dab(tex, rs, x0, y0, x1, y1, c, rough=0.3)
    y = y0 + pitch
    while y < y1 - 2:
        ruler(tex, (x0 + 1, y), (x1 - 1, y), np.clip(c * 0.62, 0, 1), 1.4, 0.8)
        ruler(tex, (x0 + 1, y + 2), (x1 - 1, y + 2), np.clip(c * 1.18, 0, 1), 1.0, 0.5)
        y += pitch * rs.uniform(0.9, 1.1)
    drips(tex, rs, int((x1 - x0) / 25), '#5a3a30', x0, x1, y_from=[y0 + 2], length=(10, (y1 - y0) * 0.6),
          alpha=(0.15, 0.4))


def windows(tex, emit, rs, x0, y0, x1, y1, cols, rows, lit_p=0.4, frame='#2a2e3a', glass='#1a2433',
            lit=('#f2e6a8', '#e8d68c', '#f6eccb', '#d9c27a'), sill='#8a8c96', gap=0.28):
    """A grid of windows, every one painted on its own."""
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows
    for r in range(rows):
        row_on = rs.random() < 0.7
        for c in range(cols):
            wx0 = x0 + c * cw + cw * gap / 2 + rs.normal(0, 0.5)
            wy0 = y0 + r * ch + ch * gap / 2 + rs.normal(0, 0.5)
            wx1, wy1 = wx0 + cw * (1 - gap), wy0 + ch * (1 - gap) * rs.uniform(0.92, 1.0)
            dab(tex, rs, wx0 - 1.2, wy0 - 1.2, wx1 + 1.2, wy1 + 1.2, frame, rough=0.4)
            if rs.random() < lit_p * (1.25 if row_on else 0.5):
                k = hx(lit[rs.integers(len(lit))]) * rs.uniform(0.85, 1.05)
                m, bb = dab(tex, rs, wx0, wy0, wx1, wy1, k, rough=0.5)
                if emit is not None and m is not None:
                    blend(emit[bb[0]:bb[1], bb[2]:bb[3]], m, k * 0.45, 1.0)
                # an interior: blind, curtain edge or a shape, each different
                kind = rs.random()
                if kind < 0.3:
                    dab(tex, rs, wx0, wy0, wx1, wy0 + (wy1 - wy0) * rs.uniform(0.2, 0.6), k * 0.72, rough=0.5)
                elif kind < 0.55:
                    xs = wx0 + (wx1 - wx0) * rs.uniform(0.1, 0.6)
                    dab(tex, rs, xs, wy0 + (wy1 - wy0) * rs.uniform(0.3, 0.6), xs + (wx1 - wx0) * rs.uniform(0.15, 0.35),
                        wy1, k * 0.55, rough=0.8)
                elif kind < 0.7:
                    dab(tex, rs, wx0 + (wx1 - wx0) * 0.5 - 0.6, wy0, wx0 + (wx1 - wx0) * 0.5 + 0.6, wy1, k * 0.5, rough=0.2)
            else:
                g = hx(glass) * rs.uniform(0.8, 1.2)
                dab(tex, rs, wx0, wy0, wx1, wy1, g, rough=0.4)
                if rs.random() < 0.5:  # a sky reflection streak across the glass
                    dab(tex, rs, wx0, wy0, wx0 + (wx1 - wx0) * 0.35, wy1, g * 1.6, alpha=0.5, rough=0.3)
                if rs.random() < 0.35:  # a drawn curtain
                    cx_ = wx0 + (wx1 - wx0) * rs.uniform(0.3, 0.7)
                    dab(tex, rs, cx_, wy0, wx1, wy1, hx(['#5a5470', '#6a5a50', '#4a5a6a'][rs.integers(3)]) * rs.uniform(.8, 1.1),
                        rough=0.5)
            if (wx1 - wx0) > 14:  # mullions on the bigger windows
                mx = (wx0 + wx1) / 2
                ruler(tex, (mx, wy0), (mx, wy1), frame, max(1.0, (wx1 - wx0) * 0.06))
                my = wy0 + (wy1 - wy0) * 0.35
                ruler(tex, (wx0, my), (wx1, my), frame, max(1.0, (wx1 - wx0) * 0.05))
            ruler(tex, (wx0 - 1, wy1 + 1.5), (wx1 + 1, wy1 + 1.5), sill, 1.6, 0.8)


def sign(w, h, text, fg='#1e1b24', bg='#efe6cf', border='#2a2430', vertical=True, size=None, font=FONT):
    """A sign panel with Japanese lettering, returned as its own texture."""
    im = Image.new('RGB', (w, h), bg)
    d = ImageDraw.Draw(im)
    d.rectangle([1, 1, w - 2, h - 2], outline=border, width=max(2, w // 30))
    if vertical:
        size = size or int(min(w * 0.7, h / max(1, len(text)) * 0.85))
        f = ImageFont.truetype(font, size)
        total = len(text) * size * 1.08
        y = (h - total) / 2
        for ch_ in text:
            bb = d.textbbox((0, 0), ch_, font=f)
            d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], y - bb[1] * 0.5), ch_, fill=fg, font=f)
            y += size * 1.08
    else:
        size = size or int(min(h * 0.62, w / max(1, len(text)) * 0.9))
        f = ImageFont.truetype(font, size)
        bb = d.textbbox((0, 0), text, font=f)
        d.text(((w - (bb[2] - bb[0])) / 2 - bb[0], (h - (bb[3] - bb[1])) / 2 - bb[1]), text, fill=fg, font=f)
    a = np.asarray(im, np.float32) / 255.0
    return cv2.GaussianBlur(a, (0, 0), 0.5)
