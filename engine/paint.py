"""Pigment texture for painted backgrounds (poster colour / gouache)."""
import cv2
import numpy as np


def _field(rs, h, w, gh, gw):
    n = rs.standard_normal((max(2, gh), max(2, gw))).astype(np.float32)
    return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)


def gouache(rgb, seed=0, amt=0.05, streak='v'):
    """Uneven pigment: brush streaks, fine tooth and broad mottling."""
    rs = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    if streak == 'v':
        s = _field(rs, h, w, h // 50, w // 3)
    else:
        s = _field(rs, h, w, h // 3, w // 50)
    fine = _field(rs, h, w, h // 2, w // 2)
    broad = _field(rs, h, w, h // 90, w // 90)
    m = 1 + amt * (0.55 * s + 0.3 * fine + 0.7 * broad)
    hue = amt * 0.4 * _field(rs, h, w, h // 120, w // 120)
    out = rgb * m[..., None]
    out[..., 2] *= 1 + hue
    out[..., 0] *= 1 - hue
    return out
