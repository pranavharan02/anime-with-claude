"""Photographed-light effects: bloom, transmitted light, ribbons, gradients."""
import cv2
import numpy as np
import skia

from .core import col, hx


def blur(a, sigma):
    if sigma <= 0:
        return a
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


def bloom(emit, levels=((3, .7), (10, .55), (30, .45), (80, .3))):
    """Emission RGB -> glow RGB (sum of gaussian halos). Downsamples big sigmas."""
    out = np.zeros_like(emit)
    h, w = emit.shape[:2]
    for sigma, amp in levels:
        if sigma > 12:
            f = int(sigma // 6)
            small = cv2.resize(emit, (w // f, h // f), interpolation=cv2.INTER_AREA)
            g = blur(small, sigma / f)
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            g = blur(emit, sigma)
        out += amp * g
    return out


def lin_grad(canvas, poly_path, p0, p1, colors, stops=None, alpha=1.0):
    """Fill a skia path with a linear gradient between points p0 and p1."""
    cs = [col(c, alpha).toColor() for c in colors]
    shader = skia.GradientShader.MakeLinear(
        [skia.Point(*p0), skia.Point(*p1)], cs, stops)
    paint = skia.Paint(AntiAlias=True, Shader=shader)
    canvas.drawPath(poly_path, paint)


def radial_grad(canvas, poly_path, c, r, colors, stops=None):
    cs = [col(x, a).toColor() for x, a in colors]
    shader = skia.GradientShader.MakeRadial(skia.Point(*c), r, cs, stops)
    canvas.drawPath(poly_path, skia.Paint(AntiAlias=True, Shader=shader))


def ribbon(poly, widths):
    """Centre polyline + per-point width -> closed outline polygon."""
    poly = np.asarray(poly, float)
    t = np.gradient(poly, axis=0)
    t /= np.hypot(t[:, 0], t[:, 1])[:, None] + 1e-9
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    w = np.asarray(widths, float)[:, None] / 2
    return np.vstack([poly + n * w, (poly - n * w)[::-1]])


def tint(rgb, c):
    return rgb * hx(c) if isinstance(c, str) else rgb * np.asarray(c)
