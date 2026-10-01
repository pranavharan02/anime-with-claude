"""Facade painting for the city, by distance class (CITY_BIBLE rules 10-15, 28-33).

Each building face is an opaque poster-paint plane: one flat tone (lit face
or shadow face), a darker top band, grime on old stock. No physical shading.
Windows change with how big a bay is on screen:
  near   (bay > 15 px)  portrait windows with a sill, lit ones cream/chartreuse
                        with dark interior shapes in the lower third
  mid    (3-15 px)      horizontal dashes 1-6 windows long, gaps, dark rows
  far    (0.8-3 px)     sponged glitter in horizontal striations
  vfar   (< 0.8 px)     no windows: a glow rising from the base
Megatowers add lit belt courses every 8-15 floors.
"""
import numpy as np

from . import gouache as G
from . import brushtex as B


def _rgb(c):
    return G.hx(c) if isinstance(c, str) else np.asarray(c, np.float32)


def window_class(bay_px):
    if bay_px > 15:
        return 'near'
    if bay_px > 1.1:
        return 'mid'
    if bay_px > 0.5:
        return 'far'
    return 'vfar'


def paint_facade(mat, wm, hm, ppm, rs, bay_px=None, lit_face=True, max_px=2600):
    """Returns (tex, emit). ppm: texture px per metre. bay_px: a bay's size on
    screen, which picks the distance class (defaults to ppm * bay / 2)."""
    ppm = float(np.clip(ppm, 0.3, max_px / max(wm, hm, 1)))
    tw, th = int(max(2, round(wm * ppm))), int(max(2, round(hm * ppm)))
    wall = _rgb(mat['wall'] if lit_face else mat['shade'])
    tex = np.empty((th, tw, 3), np.float32)
    tex[:] = wall
    emit = np.zeros_like(tex)
    # visible poster paint: strokes laid down the wall, pigment tooth, a darker base
    amp = 0.16 if ppm >= 6 else (0.11 if ppm >= 2 else 0.06)
    tex *= (1 + amp * B.sample(th, tw, ppm, rs, 'v', stroke_m=rs.uniform(0.4, 0.9)))[..., None]
    yv = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
    tex *= 1 - 0.12 * yv ** 3
    if ppm >= 2:   # sponge mottling: blotchy pigment, the way a BG painter dabs concrete
        tex *= (1 + 0.07 * G.field(rs, th, tw, max(3, ppm * 1.5), max(3, ppm * 1.5)))[..., None]
    fl, bay = mat.get('floor', 3.6), mat.get('bay', 2.6)
    rows, cols = max(1, int(round(hm / fl))), max(1, int(round(wm / bay)))
    fh, cw = th / rows, tw / cols
    cls = window_class(bay_px if bay_px is not None else bay * ppm / 2)
    pal = [_rgb(c) for c in mat.get('pal', ['#f2e2a0'])]
    lit_p = mat.get('lit', 0.5)
    shop = 1 if mat.get('shop') else 0
    # top band: a darker soffit / parapet band
    tb = max(1, int(min(th * 0.04, fh * 0.6)))
    tex[:tb] = wall * 0.72
    style = mat.get('style', 'grid')
    if style != 'none' and cls != 'vfar':
        floor_on = rs.random(rows) < 0.7
        dark_from = rs.integers(0, rows) if rows > 6 and rs.random() < 0.6 else -1
        dark_len = rs.integers(2, max(3, rows // 4)) if dark_from >= 0 else 0
        main = rs.integers(len(pal))
        for r in range(shop, rows):
            row_dark = dark_from <= r < dark_from + dark_len
            p_row = 0.0 if row_dark else (lit_p * 1.25 if floor_on[r] else lit_p * 0.4)
            y_top = th - (r + 1) * fh
            if cls == 'near':
                ry0, ry1 = y_top + fh * 0.18, y_top + fh * 0.82
                row_dy = rs.normal(0, fh * 0.03)
                ry0, ry1 = ry0 + row_dy, ry1 + row_dy
                for c in range(cols):
                    jx = rs.normal(0, cw * 0.05)
                    ww_ = cw * 0.56 * rs.uniform(0.85, 1.12)
                    rx0 = c * cw + (cw - ww_) / 2 + jx
                    rx1 = rx0 + ww_
                    sill = np.clip(wall * 1.35 + 0.03, 0, 1)
                    _rect(tex, rx0 - 1, ry1, rx1 + 1, ry1 + max(1, fh * 0.05), sill)
                    if rs.random() < p_row:
                        k = (pal[main] if rs.random() < 0.8 else pal[rs.integers(len(pal))]) * rs.uniform(0.7, 0.92)
                        _rect(tex, rx0, ry0, rx1, ry1, k)
                        _rect(emit, rx0, ry0, rx1, ry1, k * 0.3)
                        if rs.random() < 0.6:          # interior shapes in the lower third
                            sx = rx0 + (rx1 - rx0) * rs.uniform(0.0, 0.6)
                            _rect(tex, sx, ry0 + (ry1 - ry0) * rs.uniform(0.55, 0.75),
                                  min(rx1, sx + (rx1 - rx0) * rs.uniform(0.2, 0.5)), ry1, k * 0.45)
                        if rs.random() < 0.3:          # a blind half down
                            _rect(tex, rx0, ry0, rx1, ry0 + (ry1 - ry0) * rs.uniform(0.2, 0.5), k * 0.75)
                    else:
                        _rect(tex, rx0, ry0, rx1, ry1, wall * rs.uniform(0.82, 0.9))
            elif cls == 'mid':
                y_top = y_top + rs.normal(0, fh * 0.05)
                c = 0
                while c < cols:
                    run = 1 if rs.random() < 0.7 else int(rs.integers(2, 5))   # some lit runs merge
                    if rs.random() < p_row:
                        k = (pal[main] if rs.random() < 0.65 else pal[rs.integers(len(pal))]) * rs.uniform(0.55, 1.0)
                        x0_ = c * cw + cw * rs.uniform(0.15, 0.3) + rs.normal(0, cw * 0.06)
                        x1_ = (c + run) * cw - cw * rs.uniform(0.15, 0.3)
                        y0_ = y_top + fh * rs.uniform(0.25, 0.35)
                        y1_ = y_top + fh * rs.uniform(0.62, 0.75)
                        _rect(tex, x0_, y0_, x1_, y1_, k)
                        _rect(emit, x0_, y0_, x1_, y1_, k * 0.25)
                        if rs.random() < 0.3:   # a hot white-yellow core in some runs
                            hot = np.clip(k * 1.25 + 0.15, 0, 1)
                            cx_ = (x0_ + x1_) / 2
                            _rect(tex, cx_ - (x1_ - x0_) * 0.2, y0_, cx_ + (x1_ - x0_) * 0.2, y1_, hot)
                            _rect(emit, cx_ - (x1_ - x0_) * 0.2, y0_, cx_ + (x1_ - x0_) * 0.2, y1_, hot * 0.6)
                    elif rs.random() < 0.5:
                        x0_ = c * cw + cw * 0.22
                        _rect(tex, x0_, y_top + fh * 0.28, x0_ + cw * 0.55 * run, y_top + fh * 0.72,
                              wall * rs.uniform(0.78, 0.9))
                    c += run
                continue
                for c in range(cols):
                    if rs.random() < p_row:
                        k = (pal[main] if rs.random() < 0.65 else pal[rs.integers(len(pal))]) * rs.uniform(0.55, 1.0)
                        wj, hj = rs.uniform(0.75, 1.15), rs.uniform(0.8, 1.15)
                        x0_ = c * cw + cw * (1 - 0.55 * wj) / 2
                        y0_ = y_top + fh * (1 - 0.45 * hj) / 2
                        _rect(tex, x0_, y0_, x0_ + cw * 0.55 * wj, y0_ + fh * 0.45 * hj, k)
                        _rect(emit, x0_, y0_, x0_ + cw * 0.55 * wj, y0_ + fh * 0.45 * hj, k * 0.25)
                    elif rs.random() < 0.5:
                        x0_ = c * cw + cw * 0.22
                        _rect(tex, x0_, y_top + fh * 0.28, x0_ + cw * 0.55, y_top + fh * 0.72, wall * rs.uniform(0.78, 0.9))
            else:  # far: one tiny dab per lit window, never a line
                if row_dark:
                    continue
                ys_ = int(y_top + fh * 0.5)
                if 0 <= ys_ < th:
                    lit_c = np.nonzero(rs.random(cols) < p_row)[0]
                    xs = ((lit_c + 0.5) * cw).astype(int).clip(0, tw - 1)
                    k = pal[main]
                    vals = rs.uniform(0.5, 1.0, (len(xs), 1))
                    tex[ys_, xs] = k * vals
                    emit[ys_, xs] = k * vals * 0.3
    # belt courses: lit bands every n floors on megatowers
    belt = mat.get('belt', 0)
    if belt and cls != 'vfar':
        bc = _rgb(mat.get('belt_col', '#ffe0a0'))
        for r in range(belt, rows, belt):
            yy = int(th - r * fh)
            hgt = max(1, int(fh * 0.5))
            tex[max(0, yy - hgt):yy] = tex[max(0, yy - hgt):yy] * 0.45 + bc * 0.45
            emit[max(0, yy - hgt):yy] = bc * 0.08
    # shop front band on old stock
    if shop and fh >= 3:
        sh = int(fh)
        tex[th - sh:] = _rgb(mat.get('shop_col', '#1e1e28'))
        x = 0.0
        while x < tw - 2:
            wdt = rs.uniform(2.5, 6) * ppm
            if rs.random() < 0.65:
                k = _rgb(['#f0d79a', '#d8e6c0', '#ffd0a0', '#c8e0ff', '#ff9ad0'][rs.integers(5)])
                _rect(tex, x + ppm * 0.3, th - sh * 0.8, min(tw, x + wdt - ppm * 0.3), th - 1, k)
                _rect(emit, x + ppm * 0.3, th - sh * 0.8, min(tw, x + wdt - ppm * 0.3), th - 1, k * 0.4)
            x += wdt
    # grime: rust and water runs from sills on old stock
    g = mat.get('grime', 0)
    if g and ppm >= 3:
        G.drips(tex, rs, int(g * tw / 6), '#2e2420', alpha=(0.12, 0.4), length=(ppm, ppm * 7),
                y_from=[int(th - r * fh) for r in range(1, rows)] or None)
    # painted edge work: the parapet's lit top edge and a dark shadow line beneath it
    if ppm >= 1:
        lw = max(1, int(round(ppm * 0.18)))
        tex[:lw] = np.clip(wall * 1.9 + 0.06, 0, 1)
        tex[lw:2 * lw + 1] *= 0.6
        if ppm >= 2.5 and rows > 3:
            step = int(rs.choice([3, 4, 5]))
            for r in range(step, rows, step):     # string courses: light top, dark under
                yy = int(th - r * fh)
                if 2 * lw < yy < th - 2 * lw:
                    tex[yy - lw:yy] = np.clip(wall * 1.45 + 0.03, 0, 1)
                    tex[yy:yy + lw] *= 0.62

    # facade clutter: drainpipes, AC units, small signboards [9]
    if ppm >= 2.5 and style != 'none':
        for _ in range(int(rs.integers(0, 3)) + (1 if shop else 0)):
            px = rs.uniform(0.05, 0.95) * tw
            pw = max(1.0, 0.25 * ppm)
            _rect(tex, px, th * rs.uniform(0, 0.3), px + pw, th, wall * 0.55)
            _rect(tex, px, th * rs.uniform(0, 0.3), px + max(1, pw * 0.3), th, np.clip(wall * 1.5 + 0.05, 0, 1))
        if shop and ppm >= 4:
            for r in range(1, rows):
                for c in range(cols):
                    if rs.random() < 0.12:     # an AC unit under the sill
                        ux = c * cw + cw * 0.25
                        uy = th - r * fh - fh * 0.12
                        _rect(tex, ux, uy, ux + cw * 0.45, uy + fh * 0.22, _rgb('#9aa0a8') * rs.uniform(0.6, 0.9))
            for _ in range(int(rs.integers(0, 3))):    # vertical signboards on the lower floors
                sx = rs.uniform(0.05, 0.85) * tw
                sw_, sh_ = 0.8 * ppm, rs.uniform(2.5, 5) * ppm
                sy = th - fh * 1.1 - sh_
                kc = _rgb(['#efe6cf', '#f2a8c8', '#e8e0b0', '#c8e0ff', '#ff6a5a'][rs.integers(5)])
                _rect(tex, sx, sy, sx + sw_, sy + sh_, kc)
                _rect(emit, sx, sy, sx + sw_, sy + sh_, kc * 0.35)
                for gy in np.arange(sy + sw_ * 0.3, sy + sh_ - sw_ * 0.5, sw_ * 0.9):   # glyph marks
                    _rect(tex, sx + sw_ * 0.25, gy, sx + sw_ * 0.75, gy + sw_ * 0.55, kc * 0.25)

    # rim highlight on one corner, a painted 1-2 px line
    rim = mat.get('rim')
    if rim and lit_face:
        wpx = max(1, int(round(ppm * 0.25)))
        if mat.get('rim_side', 'left') == 'left':
            tex[:, :wpx] = _rgb(rim)
        else:
            tex[:, -wpx:] = _rgb(rim)
    return tex, emit


def _rect(a, x0, y0, x1, y1, col):
    h, w = a.shape[:2]
    xa, xb = int(round(x0)), int(round(x1))
    ya, yb = int(round(y0)), int(round(y1))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, max(xb, xa + 1)), min(h, max(yb, ya + 1))
    if xb > xa and yb > ya:
        a[ya:yb, xa:xb] = col
