"""The camera and lab: everything that happens when the painted cels are shot
on 35 mm film. Apply once per finished frame."""
import cv2
import numpy as np

from .core import hx
from .fx import blur


def _luma(rgb):
    return rgb[..., 0] * .299 + rgb[..., 1] * .587 + rgb[..., 2] * .114


def develop(rgb, frame=0, seed=0, grain=0.035, weave=0.45, soft=0.55,
            halation=0.22, lift='#0a0f1c', lift_amt=0.045, vignette=0.22,
            dust=1.0):
    rs = np.random.default_rng(seed * 100003 + frame)
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32).copy()

    # Gate weave: the film is never registered perfectly in the gate.
    if weave:
        wr = np.random.default_rng(seed)
        ph = wr.uniform(0, 6.28, 4)
        dx = weave * (0.6 * np.sin(frame * 0.71 + ph[0]) + 0.4 * np.sin(frame * 2.3 + ph[1]))
        dy = weave * (0.6 * np.sin(frame * 0.53 + ph[2]) + 0.4 * np.sin(frame * 1.9 + ph[3]))
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)

    # Halation: bright light bleeds red-orange through the emulsion.
    if halation:
        hi = np.clip(_luma(img) - 0.7, 0, None)[..., None] * img
        img += halation * blur(hi, 9) * hx('#ff6a3a') * 1.6

    if soft:
        img = blur(img, soft)

    # Film shoulder: highlights roll off instead of clipping.
    img = np.where(img < 0.82, img, 0.82 + 0.18 * np.tanh((img - 0.82) / 0.18))

    # Blacks lift toward a cool base (the print never reaches true black).
    base = hx(lift)
    img = base * lift_amt + img * (1 - lift_amt)

    if vignette:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / 1.414
        img *= (1 - vignette * r ** 2.2)[..., None]

    if grain:
        g = rs.standard_normal((h, w, 1)).astype(np.float32)
        c = rs.standard_normal((h, w, 3)).astype(np.float32) * 0.35
        n = blur(g + c, 0.7)
        L = _luma(np.clip(img, 0, 1))[..., None]
        img += grain * n * (0.45 + 2.2 * L * (1 - L))

    if dust:
        _dust(img, rs, dust)

    return np.clip(img, 0, 1)


def _dust(img, rs, amount):
    h, w = img.shape[:2]
    n = rs.poisson(2.2 * amount)
    for _ in range(n):
        x, y = rs.uniform(0, w), rs.uniform(0, h)
        r = rs.uniform(0.8, 2.6)
        bright = rs.random() < 0.6  # dirt on the negative prints white
        val = 0.85 if bright else 0.04
        mask = np.zeros((h, w), np.float32)
        if rs.random() < 0.15:  # a hair
            pts = [(x, y)]
            ang = rs.uniform(0, 6.28)
            for _ in range(rs.integers(6, 14)):
                ang += rs.normal(0, 0.5)
                x += np.cos(ang) * rs.uniform(4, 9)
                y += np.sin(ang) * rs.uniform(4, 9)
                pts.append((x, y))
            cv2.polylines(mask, [np.int32(pts)], False, 1.0, 1, cv2.LINE_AA)
        else:
            cv2.ellipse(mask, (int(x), int(y)), (int(r + 1), int(r * rs.uniform(.5, 1) + 1)),
                        rs.uniform(0, 180), 0, 360, 1.0, -1, cv2.LINE_AA)
        mask = blur(mask, 0.6)[..., None] * rs.uniform(0.35, 0.8)
        img[:] = img * (1 - mask) + val * mask
