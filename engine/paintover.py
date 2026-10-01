"""Paint-over: repaint a layout render in flat brush strokes.

A background painter works from a layout: the drawing fixes where things go,
the paint is laid by hand. Here the 3D render is the layout and colour guide,
and this module lays the paint: a broad underpainting, then flat strokes in
decreasing sizes (after Hertzmann's layered painterly rendering), placed only
where the canvas still differs from the guide. Strokes follow each surface's
direction and are clipped to the surface that owns them (the stage's ID map),
so architectural edges stay crisp while every area becomes brushwork.
"""
import cv2
import numpy as np


def paint_over(ref, ids, dirs, seed=0, widths=(8.0, 4.5, 2.2), thresh=(0.05, 0.035, 0.025),
               length=(2.0, 5.0), sky_dir=0.0, var=0.025, ss=2):
    rs = np.random.default_rng(seed)
    h, w = ref.shape[:2]
    H, W = h * ss, w * ss
    guide = cv2.resize(ref, (W, H), interpolation=cv2.INTER_LINEAR)
    idm = cv2.resize(ids, (W, H), interpolation=cv2.INTER_NEAREST)
    canvas = cv2.GaussianBlur(guide, (0, 0), 10 * ss)          # underpainting: a broad wash
    # the wash must not bleed across surfaces either: per-surface mean where possible
    for i in np.unique(idm):
        m = idm == i
        if m.sum() > 50:
            canvas[m] = cv2.GaussianBlur(np.where(m[..., None], guide, 0), (0, 0), 6 * ss)[m] / np.maximum(
                cv2.GaussianBlur(m.astype(np.float32), (0, 0), 6 * ss)[m][:, None], 1e-3)
    for bw, T in zip(widths, thresh):
        bw2 = bw * ss
        blur = cv2.GaussianBlur(guide, (0, 0), max(0.6, bw2 * 0.45))
        err = cv2.blur(np.abs(blur - canvas).mean(2), (int(bw2) | 1, int(bw2) | 1))
        step = max(1.0, bw2 * 0.85)
        gy, gx = np.mgrid[0:H:step, 0:W:step]
        gx = (gx + rs.uniform(0, step, gx.shape)).astype(int).ravel()
        gy = (gy + rs.uniform(0, step, gy.shape)).astype(int).ravel()
        keep = (gx < W) & (gy < H)
        gx, gy = gx[keep], gy[keep]
        sel = err[gy, gx] > T
        gx, gy = gx[sel], gy[sel]
        order = rs.permutation(len(gx))
        colbuf = np.zeros_like(canvas)
        idbuf = np.full((H, W), -9, np.int32)
        th = max(1, int(round(bw2)))
        for k in order:
            x, y = int(gx[k]), int(gy[k])
            sid = int(idm[y, x])
            ang = dirs.get(sid, sky_dir) + rs.normal(0, 0.04)
            L = bw2 * rs.uniform(*length) / 2
            dx, dy = np.cos(ang) * L, np.sin(ang) * L
            c = blur[y, x] * (1 + rs.normal(0, var))
            p0, p1 = (int(x - dx), int(y - dy)), (int(x + dx), int(y + dy))
            cv2.line(colbuf, p0, p1, tuple(float(v) for v in c), th, cv2.LINE_8)
            cv2.line(idbuf, p0, p1, sid, th, cv2.LINE_8)
        ok = idbuf == idm                     # a stroke only lands on its own surface
        canvas[ok] = colbuf[ok]
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
