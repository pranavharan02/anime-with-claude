"""Turn a reference frame into code: a layered cel scene.

Runs a cel production in reverse and writes each layer as data you can read
and edit:

  airbrush  a gradient mesh (triangles with per-corner colours) for the soft,
            airbrushed parts of the painting: skies, clouds, glows
  paint     flat cel regions (or linear gradients) traced as sub-pixel paths,
            stacked largest-first like painter's order
  ink       the traced lines: centrelines with per-point width and colour,
            redrawn as strokes on top

Pipeline:
  1. find thin dark ink with a morphological black-hat, skeletonise it,
     trace the skeleton into polylines
  2. inpaint the ink away so the paint layer sees clean colour fields
  3. flatten grain (mean-shift), find the palette (k-means in Lab), clean
     specks, split into connected regions, trace outlines (marching squares)
  4. regions whose edges are soft in the original go to the airbrush mesh;
     hard-edged regions stay as cel paths
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize


# ---------------------------------------------------------------- palette ----
def _lab(img):
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def palette_labels(img, k, seed=0, sample=250_000):
    lab = _lab(img).reshape(-1, 3)
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(lab), min(sample, len(lab)), replace=False)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.05)
    cv2.setRNGSeed(seed)
    _, _, centers = cv2.kmeans(lab[idx].astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = np.empty(len(lab), np.int32)
    for s in range(0, len(lab), 150_000):
        d = ((lab[s:s + 150_000, None, :] - centers[None]) ** 2).sum(-1)
        labels[s:s + 150_000] = d.argmin(1)
    return labels.reshape(img.shape[:2])


def clean_labels(labels, min_area):
    """Specks smaller than min_area take the label of the nearest big region."""
    keep = np.zeros(labels.shape, bool)
    for lab in np.unique(labels):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == lab).astype(np.uint8), connectivity=4)
        big = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area)[0]
        big = big[big != 0]
        keep |= np.isin(cc, big)
    _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
    return labels[iy, ix]


def majority(labels, iters=2, size=3):
    """Mode filter: slivers (JPEG ringing, edge fringes) join the majority.
    Safe on the paint layer because the ink already lives in its own layer."""
    labs = np.unique(labels)
    for _ in range(iters):
        best = np.full(labels.shape, -1.0, np.float32)
        out = labels.copy()
        for lab in labs:
            cnt = cv2.boxFilter((labels == lab).astype(np.float32), -1, (size, size), normalize=False)
            cnt += (labels == lab) * 0.5   # ties keep the current label
            win = cnt > best
            out[win] = lab
            best[win] = cnt[win]
        labels = out
    return labels


# -------------------------------------------------------------------- ink ----
def ink_mask(img, thresh=22, size=7, min_px=8):
    L = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    L = cv2.bilateralFilter(L, 5, 25, 3)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    bh = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, k)
    m = (bh > thresh).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    good = np.where(stats[:, cv2.CC_STAT_AREA] >= min_px)[0]
    good = good[good != 0]
    return np.isin(cc, good)


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(sk):
    """Skeleton pixels -> list of (N, 2) [x, y] polylines, split at junctions."""
    h, w = sk.shape
    pad = np.pad(sk, 1)
    ys, xs = np.nonzero(pad)
    nbrs = {}
    for y, x in zip(ys, xs):
        nbrs[(y, x)] = [(y + dy, x + dx) for dy, dx in _NB if pad[y + dy, x + dx]]
    seen = set()
    paths = []

    def walk(a, b):
        path = [a, b]
        seen.add((a, b)); seen.add((b, a))
        prev, cur = a, b
        while len(nbrs[cur]) == 2:
            nxt = nbrs[cur][0] if nbrs[cur][1] == prev else nbrs[cur][1]
            if (cur, nxt) in seen:
                break
            seen.add((cur, nxt)); seen.add((nxt, cur))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for p, ns in nbrs.items():
        if len(ns) != 2:
            if not ns:
                paths.append([p])
            for q in ns:
                if (p, q) not in seen:
                    paths.append(walk(p, q))
    for p, ns in nbrs.items():      # closed loops with no junctions
        for q in ns:
            if (p, q) not in seen:
                paths.append(walk(p, q))
    out = []
    for path in paths:
        a = np.array(path, float)[:, ::-1] - 1  # (y, x) padded -> (x, y)
        out.append(a)
    return out


def ink_layer(img, mask):
    sk = skeletonize(mask)
    dist = ndimage.distance_transform_edt(mask)
    lines = []
    for p in trace_skeleton(sk):
        xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
        wid = np.clip(2 * dist[yi, xi] - 0.6, 0.8, 6)
        colr = np.median(img[yi, xi].astype(np.float32), 0)
        if len(p) >= 5:  # smooth the pixel staircase, keep the ends
            sm = p.copy()
            sm[1:-1] = (p[:-2] + p[1:-1] * 2 + p[2:]) / 4
            p = sm
            wid = np.convolve(np.pad(wid, 1, mode='edge'), [1 / 3] * 3, 'valid')
        step = 2 if len(p) > 6 else 1
        keep = np.r_[np.arange(0, len(p) - 1, step), len(p) - 1]
        lines.append((p[keep] + 0.5, wid[keep], colr, (yi, xi)))
    return lines


# ------------------------------------------------------------------ paint ----
def _fit_fill(img, ys, xs):
    px = img[ys, xs].astype(np.float32)
    mean = px.mean(0)
    if len(xs) < 200:
        return ('flat', mean)
    if len(xs) > 6000:
        sel = np.random.default_rng(0).choice(len(xs), 6000, replace=False)
        ys, xs, px = ys[sel], xs[sel], px[sel]
    A = np.stack([np.ones_like(xs, np.float32), xs.astype(np.float32), ys.astype(np.float32)], 1)
    coef, *_ = np.linalg.lstsq(A, px, rcond=None)
    flat_err = ((px - mean) ** 2).mean()
    lin_err = ((px - A @ coef) ** 2).mean()
    lum = coef[1:, :] @ np.array([.299, .587, .114], np.float32)
    g = np.linalg.norm(lum)
    if flat_err < 3 or lin_err > 0.7 * flat_err or g < 1e-3:
        return ('flat', mean)
    u = lum / g
    t = xs * u[0] + ys * u[1]
    cx, cy = xs.mean(), ys.mean()
    tc = cx * u[0] + cy * u[1]
    p0 = np.array([cx, cy]) + u * (t.min() - tc)
    p1 = np.array([cx, cy]) + u * (t.max() - tc)
    ev = lambda p: np.clip(coef[0] + coef[1] * p[0] + coef[2] * p[1], 0, 255)
    return ('lin', p0, p1, ev(p0), ev(p1))


def _poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def mesh_layer(img, soft, step):
    """Gradient mesh sampled from the soft pixels only (normalised convolution)."""
    h, w = soft.shape
    f = img.astype(np.float32)
    s = soft.astype(np.float32)
    sig = step * 0.55
    num = cv2.GaussianBlur(f * s[..., None], (0, 0), sig)
    den = cv2.GaussianBlur(s, (0, 0), sig)[..., None]
    plain = cv2.GaussianBlur(f, (0, 0), sig)
    field = np.where(den > 0.05, num / np.maximum(den, 1e-6), plain)
    cols = int(np.ceil((w - 1) / step)) + 1
    rows = int(np.ceil((h - 1) / step)) + 1
    gx = np.minimum(np.arange(cols) * step, w - 1)
    gy = np.minimum(np.arange(rows) * step, h - 1)
    colors = field[gy[:, None], gx[None, :]]
    return dict(step=step, cols=cols, rows=rows, colors=colors)


def vectorize(img, k=64, min_area=6, eps=0.4, denoise=(3, 8), ink_thresh=22,
              soft_min=1.5, lens=0.7, mesh_step=12):
    ink = ink_mask(img, ink_thresh) if ink_thresh else np.zeros(img.shape[:2], bool)
    lines = ink_layer(img, ink) if ink.any() else []
    clean = cv2.inpaint(np.ascontiguousarray(img), cv2.dilate(ink.astype(np.uint8), np.ones((3, 3))),
                        3, cv2.INPAINT_TELEA) if ink.any() else img
    flat = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(clean[..., ::-1]), *denoise)[..., ::-1]
    labels = clean_labels(majority(palette_labels(flat, k)), min_area)

    # Edge profile in perceptual colour (delta-E per pixel) at two scales.
    # For a step edge blurred by sigma s, the gradient peak falls as
    # 1/sqrt(s^2 + k^2) for a measuring blur k, so the ratio of the fine
    # (k=0.8) and coarse (k=3) responses gives s. Cel edges come out near
    # zero, airbrush edges wide. That width becomes the region's feather.
    lab = _lab(flat)
    def egrad(sig):
        b_ = cv2.GaussianBlur(lab, (0, 0), sig)
        gx = cv2.Sobel(b_, cv2.CV_32F, 1, 0, ksize=3) / 8
        gy = cv2.Sobel(b_, cv2.CV_32F, 0, 1, ksize=3) / 8
        return np.sqrt((gx ** 2 + gy ** 2).sum(-1))
    fine, coarse = egrad(0.8), egrad(3.0)

    fills = []
    for lab_id in np.unique(labels):
        n, cc, stats, _ = cv2.connectedComponentsWithStats((labels == lab_id).astype(np.uint8), connectivity=4)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(labels.shape[1], x + w + 2), min(labels.shape[0], y + h + 2)
            reg = cc[y0:y1, x0:x1] == i
            ring = cv2.dilate(reg.astype(np.uint8), np.ones((3, 3))).astype(bool) & ~reg
            feather = 0.0
            if ring.any():
                ef = float(fine[y0:y1, x0:x1][ring].mean())
                ec = float(coarse[y0:y1, x0:x1][ring].mean())
                r2 = (ef / max(ec, 1e-4)) ** 2
                # effective measuring blurs include Sobel's own (~0.7 px)
                s2 = (9.5 - 1.13 * r2) / (r2 - 1) if r2 > 1.0001 else 144.0
                feather = float(np.sqrt(max(0.0, min(s2, 144.0) - lens ** 2)))
                if feather < soft_min:
                    feather = 0.0
            filled = ndimage.binary_fill_holes(reg)
            cs = measure.find_contours(np.pad(filled, 1).astype(np.float32), 0.5)
            if not cs:
                continue
            c = max(cs, key=len)
            pts = np.stack([c[:, 1] - 1 + x0, c[:, 0] - 1 + y0], 1).astype(np.float32)
            pts = cv2.approxPolyDP(pts.reshape(-1, 1, 2), eps + 0.15 * feather, True).reshape(-1, 2) + 0.5
            if len(pts) < 3:
                continue
            if feather:  # a thin region would vanish under its own feather
                perim = float(np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T).sum())
                feather = min(feather, 2 * area / max(perim, 1) / 3, 8.0)
                if feather < soft_min:
                    feather = 0.0
            ys, xs = np.nonzero(reg)
            core = cv2.erode(reg.astype(np.uint8), np.ones((3, 3))).astype(bool)
            cy_, cx_ = np.nonzero(core if core.sum() >= 3 else reg)
            fills.append((_poly_area(pts), _fit_fill(clean, ys + y0, xs + x0), pts, feather,
                          (cy_ + y0, cx_ + x0)))
    fills.sort(key=lambda s: -s[0])
    mesh = mesh_layer(clean, np.ones(labels.shape, bool), mesh_step)
    soft_frac = sum(f[0] for f in fills if f[3] > 0) / float(labels.size)
    return dict(w=img.shape[1], h=img.shape[0], mesh=mesh, fills=fills, lines=lines,
                soft_frac=float(min(1.0, soft_frac)))
