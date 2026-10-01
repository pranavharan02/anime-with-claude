"""Gouache brushwork tiles: the visible paint on every surface.

A tile is built from a few thousand overlapping semi-opaque flat strokes
(mostly running one way, as a background painter lays a wall in), with soft
edges and pigment granulation. Faces sample a crop at a scale matched to how
big a brush stroke is in metres, so near walls show strokes and far ones a
fine tooth. Values are a zero-mean modulation field (std ~1).
"""
import cv2
import numpy as np

_TILES = {}


def tile(direction='v', size=1024, seed=0):
    key = (direction, size, seed)
    if key in _TILES:
        return _TILES[key]
    rs = np.random.default_rng(seed)
    acc = np.zeros((size, size), np.float32)
    for _ in range(3200):
        L = rs.uniform(40, 220)
        w = rs.uniform(5, 22)
        ang = (np.pi / 2 if direction == 'v' else 0.0) + rs.normal(0, 0.08)
        cx, cy = rs.uniform(0, size), rs.uniform(0, size)
        dx, dy = np.cos(ang) * L / 2, np.sin(ang) * L / 2
        m = np.zeros((size, size), np.float32)
        cv2.line(m, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), 1.0, int(w), cv2.LINE_AA)
        x0, x1 = int(max(0, min(cx - dx, cx + dx) - w)), int(min(size, max(cx - dx, cx + dx) + w))
        y0, y1 = int(max(0, min(cy - dy, cy + dy) - w)), int(min(size, max(cy - dy, cy + dy) + w))
        if x1 <= x0 or y1 <= y0:
            continue
        a = rs.uniform(0.25, 0.55)
        v = rs.normal(0, 1)
        sub = m[y0:y1, x0:x1]
        acc[y0:y1, x0:x1] = acc[y0:y1, x0:x1] * (1 - a * sub) + v * a * sub
    acc = cv2.GaussianBlur(acc, (0, 0), 0.8)
    gran = cv2.GaussianBlur(rs.standard_normal((size, size)).astype(np.float32), (0, 0), 0.9)
    acc = acc / (acc.std() + 1e-6) + 0.35 * gran / (gran.std() + 1e-6)
    acc /= acc.std() + 1e-6
    _TILES[key] = acc
    return acc


def sample(th, tw, ppm, rs, direction='v', stroke_m=0.6):
    """A (th, tw) crop with strokes about stroke_m metres wide at ppm px/m."""
    t = tile(direction)
    # a tile stroke is ~13 px wide; scale so it covers stroke_m metres
    scale = (stroke_m * ppm) / 13.0
    need_h, need_w = int(th / max(scale, 1e-3)) + 2, int(tw / max(scale, 1e-3)) + 2
    S = t.shape[0]
    if need_h >= S or need_w >= S:
        crop = cv2.resize(t, (min(need_w, 4 * S), min(need_h, 4 * S)), interpolation=cv2.INTER_AREA) \
            if (need_h > 4 * S or need_w > 4 * S) else np.tile(t, (need_h // S + 1, need_w // S + 1))[:need_h, :need_w]
    else:
        y, x = rs.integers(0, S - need_h), rs.integers(0, S - need_w)
        crop = t[y:y + need_h, x:x + need_w]
    out = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    if scale < 1:   # downsampled: the strokes average out, renormalise toward a fine tooth
        out = out / (out.std() + 1e-6) * min(1.0, 0.4 + scale)
    return out
