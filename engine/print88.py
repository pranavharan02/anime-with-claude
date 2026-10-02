"""Shoot and print: what the camera stand and the 1988 release print add.

Grain is synthesised to the measurements taken from real frames (flat areas,
high-passed at sigma 3): about 1.5-2.5 levels std, ~0.95 correlated across
channels, neighbour correlation 0.65-0.8 (clumps of 1-2 px) with some coarser
mottling on top.
"""
import cv2
import numpy as np


def _n(rs, h, w, sig, ch=1):
    n = rs.standard_normal((h, w, ch)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sig)
    if n.ndim == 2:
        n = n[..., None]
    return n / (n.std() + 1e-6)


def bloom(emit, levels=((1.2, .5), (3.5, .28), (9, .1))):
    out = np.zeros_like(emit)
    for s, a in levels:
        out += a * cv2.GaussianBlur(emit, (0, 0), s)
    return out


_BANK = None


def _bank_available(bank_path=None):
    from pathlib import Path
    bp = bank_path or Path(__file__).resolve().parents[1] / 'out' / 'texbank.npz'
    return _BANK is not None or Path(bp).exists()


def film_texture(img, rs, bank_path=None):
    """Quilt real print texture (paint tooth + grain) over the frame, picking
    patches whose source brightness matches the local brightness."""
    global _BANK
    if _BANK is None:
        from pathlib import Path
        bp = bank_path or Path(__file__).resolve().parents[1] / 'out' / 'texbank.npz'
        d = np.load(bp)
        _BANK = (d['patches'], d['levels'])
    patches, levels = _BANK
    P = patches.shape[1]
    st = P // 2
    h, w = img.shape[:2]
    L = cv2.GaussianBlur(img @ np.array([.299, .587, .114], np.float32), (0, 0), 6)
    win = np.outer(np.hanning(P), np.hanning(P)).astype(np.float32)[..., None] + 1e-3
    acc = np.zeros((h + 2 * P, w + 2 * P, 3), np.float32)
    wsq = np.zeros((h + 2 * P, w + 2 * P, 1), np.float32)
    order = np.argsort(levels)
    for y in range(-st, h, st):
        for x in range(-st, w, st):
            lv = L[min(max(y + st, 0), h - 1), min(max(x + st, 0), w - 1)]
            i = np.searchsorted(levels[order], lv)
            j = order[np.clip(i + rs.integers(-40, 40), 0, len(order) - 1)]
            pt = np.rot90(patches[j], rs.integers(4))
            if rs.random() < 0.5:
                pt = pt[:, ::-1]
            ys, xs = y + st, x + st
            acc[ys:ys + P, xs:xs + P] += pt * win
            wsq[ys:ys + P, xs:xs + P] += win ** 2
    tex = acc / np.sqrt(wsq + 1e-6)
    return tex[st:st + h, st:st + w] / 255.0


def wobble(img, rs, amp=0.7, scale=40):
    """Hand irregularity: nothing in a painting is ruled perfectly."""
    h, w = img.shape[:2]
    def f():
        n = rs.standard_normal((h // scale + 2, w // scale + 2)).astype(np.float32)
        return cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    fine = lambda: cv2.resize(rs.standard_normal((h // 6 + 2, w // 6 + 2)).astype(np.float32), (w, h),
                              interpolation=cv2.INTER_CUBIC) * amp * 0.35
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, mx + f() + fine(), my + f() + fine(), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def develop(img, emit=None, seed=0, lens=0.7, grain=(1.7, 0.9), halation=0.35, lift=(0.012, 0.016, 0.024),
            dust=4, bloom_amt=1.0, texbank=False):
    rs = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.astype(np.float32).copy()
    if emit is not None:
        out += bloom_amt * bloom(emit)
    L = out @ np.array([.299, .587, .114], np.float32)
    hi = np.clip(L - 0.6, 0, None)[..., None] * out
    ring = np.clip(cv2.GaussianBlur(hi, (0, 0), 7) - 0.6 * cv2.GaussianBlur(hi, (0, 0), 3), 0, None)
    out += halation * ring * np.array([1.0, 0.45, 0.3], np.float32)       # a bounded red halation ring
    out = np.where(out < 0.85, out, 0.85 + 0.15 * np.tanh((out - 0.85) / 0.15))
    if lens:
        out = cv2.GaussianBlur(out, (0, 0), lens)
    out = np.array(lift, np.float32) + out * (1 - np.array(lift, np.float32))
    if texbank and not _bank_available():
        texbank = False      # no local texture bank (tools/texbank.py): fall back to synthetic grain
    if texbank:
        out += film_texture(np.clip(out, 0, 1), rs)
        out += (_n(rs, h, w, 4.0) * 0.45 / 255.0)          # the print's faint coarse mottle (measured)
        grain = (0.0, 0.0)    # film texture only; a uniform overlay read as digital noise (round c03)
    # grain: luminance-dominant, fine clumps plus coarser mottle, scaled by level
    fine = _n(rs, h, w, 0.75)
    coarse = _n(rs, h, w, 2.2)
    chroma = _n(rs, h, w, 0.8, 3)
    g = (grain[0] * fine + grain[1] * coarse) + 0.3 * grain[0] * chroma
    Lc = np.clip(out @ np.array([.299, .587, .114], np.float32), 0, 1)[..., None]
    out += g / 255.0 * (0.55 + 0.9 * np.sqrt(Lc))
    for _ in range(rs.poisson(dust)):
        x, y = int(rs.uniform(0, w)), int(rs.uniform(0, h))
        m = np.zeros((h, w), np.float32)
        if rs.random() < 0.12:
            pts = [(x, y)]
            a = rs.uniform(0, 6.3)
            for _ in range(rs.integers(5, 12)):
                a += rs.normal(0, 0.5)
                x, y = x + 4 * np.cos(a), y + 4 * np.sin(a)
                pts.append((x, y))
            cv2.polylines(m, [np.int32(pts)], False, 1.0, 1, cv2.LINE_AA)
        else:
            cv2.circle(m, (x, y), int(rs.uniform(1, 2.5)), 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 0.5)[..., None] * rs.uniform(0.25, 0.6)
        val = 0.8 if rs.random() < 0.35 else 0.03
        out = out * (1 - m) + val * m
    return np.clip(out, 0, 1)
