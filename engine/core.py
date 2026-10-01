"""Frame buffers, colour helpers and compositing.

Layers are float32 premultiplied RGBA arrays in 0..1, shape (H, W, 4).
Skia draws vectors into surfaces; numpy does all compositing and film work.
"""
import numpy as np
import skia
from PIL import Image

W, H = 1920, 1038  # 1.85:1, the film's projection ratio


def surface(w=W, h=H):
    info = skia.ImageInfo.Make(w, h, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
    return skia.Surface.MakeRaster(info)


def to_np(surf):
    """Surface -> premultiplied float RGBA. (skia's toarray() un-premultiplies.)"""
    a = surf.makeImageSnapshot().toarray().astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    return a


def blank(w=W, h=H):
    return np.zeros((h, w, 4), np.float32)


def hx(h):
    """'#rrggbb' -> np.array([r, g, b]) in 0..1."""
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def col(c, a=1.0):
    """Hex string or rgb array -> skia.Color4f."""
    if isinstance(c, str):
        c = hx(c)
    return skia.Color4f(float(c[0]), float(c[1]), float(c[2]), float(a))


def mix(a, b, t):
    a = hx(a) if isinstance(a, str) else np.asarray(a, np.float32)
    b = hx(b) if isinstance(b, str) else np.asarray(b, np.float32)
    return a + (b - a) * t


def over(dst, src):
    """Porter-Duff source-over on premultiplied layers."""
    return src + dst * (1.0 - src[..., 3:4])


def add(dst, rgb):
    out = dst.copy()
    out[..., :3] += rgb
    return out


def screen(dst, rgb):
    out = dst.copy()
    out[..., :3] = 1.0 - (1.0 - out[..., :3]) * (1.0 - np.clip(rgb, 0, 1))
    return out


def flatten(layer, bg=(0, 0, 0)):
    """Premultiplied RGBA -> opaque RGB."""
    return layer[..., :3] + np.asarray(bg, np.float32) * (1.0 - layer[..., 3:4])


def save(rgb, path):
    a = np.clip(rgb[..., :3], 0, 1)
    Image.fromarray((a * 255 + 0.5).astype(np.uint8)).save(path)


def rng(seed):
    return np.random.default_rng(seed)
