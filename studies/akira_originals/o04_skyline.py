"""Original 04: "Skyline". A megacity at night, looking up between towers.

Vocabulary studied from the film's night-city shots: towers lit from below by
the city glow, windows as bands of tiny dashes (not big cells), bright rim
lines on the corners, hard searchlight beams, red beacons on every roof,
saturated magenta/orange far towers behind dark teal near ones.
Original composition, no film elements.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from engine import gouache as G
from engine.stage import Camera, Stage
from engine.print88 import develop, wobble

OUT = ROOT / 'out' / 'akira_originals'
W, H = 1280, 718


def tower_tex(rs, wm, hm, top, bot, lit, rim=None, style='dash', ppm=3.0, floor_m=3.6, lit_p=0.5,
              rim_side='left', bands=None):
    """A tower face painted flat: glow-lit gradient, window bands, rim line."""
    tw, th = int(max(16, wm * ppm)), int(max(16, hm * ppm))
    y = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
    t = G.hx(top) * (1 - y) + G.hx(bot) * y
    t = np.broadcast_to(t, (th, tw, 3)).copy()
    t *= (1 + 0.03 * G.field(rs, th, tw, th / 2, 8))[..., None]
    e = np.zeros_like(t)
    fh = floor_m * ppm
    rows = int(hm / floor_m)
    cols = int(wm / 2.4)
    cw = tw / max(cols, 1)
    on_floor = rs.random(rows) < 0.75
    for r in range(rows):
        y0 = r * fh + fh * 0.3
        y1 = y0 + fh * (0.38 if style == 'dash' else 0.5)
        if y1 > th - 2:
            break
        if style == 'band':
            x = 0
            while x < tw:
                run = rs.uniform(cw * 2, cw * 9)
                if rs.random() < lit_p * (1.2 if on_floor[r] else 0.4):
                    k = G.hx(lit[rs.integers(len(lit))]) * rs.uniform(0.75, 1.05)
                    G.dab(t, rs, x, y0, min(tw, x + run), y1, k, rough=0.3)
                    G.blend(e[int(y0):int(y1) + 1, int(x):int(min(tw, x + run))],
                            np.ones((int(y1) + 1 - int(y0), int(min(tw, x + run)) - int(x)), np.float32), k * 0.35)
                x += run + rs.uniform(0, cw)
            continue
        for c in range(cols):
            x0 = c * cw + cw * 0.22
            x1 = x0 + cw * 0.5
            if rs.random() < lit_p * (1.25 if on_floor[r] else 0.35):
                k = G.hx(lit[rs.integers(len(lit))]) * rs.uniform(0.7, 1.05)
                G.dab(t, rs, x0, y0, x1, y1, k, rough=0.25, var=0.08)
                e[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = k * 0.32
            else:
                G.dab(t, rs, x0, y0, x1, y1, t[int(y0), int(x0)] * 0.6, rough=0.2)
    for yy in (bands or []):  # lit setback bands
        G.ruler(t, (0, yy * th), (tw, yy * th), '#ff9ad0', 2, 0.9)
        e[max(0, int(yy * th) - 1):int(yy * th) + 2] += G.hx('#ff6ab0') * 0.4
    if rim:
        x = 1 if rim_side == 'left' else tw - 2
        G.ruler(t, (x, 0), (x, th), rim, 2.2, 0.95)
        e[:, max(0, x - 1):x + 2] += G.hx(rim) * 0.25
    return t, e


def prism_faces(cx, cz, r, n, y0, y1):
    """Side faces of an n-gon prism (a cylinder tower), as quads."""
    out = []
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        p0 = (cx + r * np.cos(a0), cz + r * np.sin(a0))
        p1 = (cx + r * np.cos(a1), cz + r * np.sin(a1))
        out.append(((p0[0], y1, p0[1]), (p1[0], y1, p1[1]), (p1[0], y0, p1[1]), (p0[0], y0, p0[1])))
    return out


def backdrop(rs):
    """The far city: glowing towers in magenta and orange haze, painted flat,
    searchlights cutting up through them."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(yy / H, 0, 1)[..., None]
    img = G.hx('#1a0612') * (1 - t) + G.hx('#6a1a3a') * t
    img += np.exp(-(((xx - 760) / 520) ** 2 + ((yy - 520) / 320) ** 2))[..., None] * G.hx('#b8483a') * 0.6
    emit = np.zeros_like(img)
    for lay, (n, hmin, hmax, wmin, wmax, top, bot, lit) in enumerate((
            (16, 260, 640, 70, 150, '#5a1a3a', '#b0503a', ('#ffcf7a', '#ffb070', '#ffe0a0')),
            (12, 200, 520, 90, 180, '#3a1630', '#8a3a4a', ('#ffd890', '#ff9ac8', '#ffe8b0')))):
        for _ in range(n):
            w = rs.uniform(wmin, wmax)
            x = rs.uniform(-60, W)
            h = rs.uniform(hmin, hmax)
            base = H + 10
            tex, e = tower_tex(rs, w / 3, h / 3, top, bot, lit, style=['band', 'dash'][rs.integers(2)], ppm=3,
                               floor_m=3.0, lit_p=0.55)
            tex = cv2.resize(tex, (int(w), int(h)))
            e = cv2.resize(e, (int(w), int(h)))
            y0 = int(base - h)
            x0 = int(x)
            xa, xb = max(0, x0), min(W, x0 + int(w))
            ya = max(0, y0)
            if xb <= xa:
                continue
            haze = 0.35 if lay == 0 else 0.15
            img[ya:base, xa:xb] = tex[ya - y0:, xa - x0:xb - x0][:H - ya] * (1 - haze) + G.hx('#a8405a') * haze
            emit[ya:base, xa:xb] += e[ya - y0:, xa - x0:xb - x0][:H - ya] * (1 - haze)
    # searchlights: hard-edged beams, additive
    for (x0, x1, spread, col) in ((300, 120, 7, '#e8f4ff'), (620, 760, 6, '#bff0ff'), (980, 1120, 8, '#e0e8ff'),
                                  (1150, 1040, 5, '#9fe8ff')):
        beam = np.zeros((H, W), np.float32)
        pts = np.array([[x0 - 4, H], [x0 + 4, H], [x1 + spread, 0], [x1 - spread, 0]], np.int32)
        cv2.fillPoly(beam, [pts], 1.0, cv2.LINE_AA)
        beam *= np.clip(1 - yy / H * 0.3, 0, 1)
        img += beam[..., None] * G.hx(col) * 0.55
        emit += beam[..., None] * G.hx(col) * 0.25
    return np.clip(img, 0, 1.5), emit


def build(rs):
    cam = Camera(W, H, fov_deg=50, eye=(0, 70, 0), pitch=5)
    st = Stage(cam)
    near_pal = ('#cfeee6', '#f4f2cc', '#a8dcff', '#ffe0a0', '#e8f8f0')
    beacons = []
    # a field of towers: nearer ones darker and teal, all with tops in frame
    towers = []
    for i in range(34):
        z = rs.uniform(380, 1100)
        x = rs.uniform(-0.62, 0.62) * z
        h = rs.uniform(110, 330) * (0.8 + 0.5 * (z / 1100))
        w = rs.uniform(40, 90)
        kind = rs.choice(['box', 'box', 'step', 'cyl'])
        towers.append((x, z, w, w * rs.uniform(0.6, 1.0), h, kind))
    for x, z, w, d, h, kind in sorted(towers, key=lambda t: -t[1]):
        dk = (z - 380) / 720                       # 0 near .. 1 far
        top = G.hx('#0a1720') * (1 - dk) + G.hx('#2a2440') * dk
        bot = G.hx('#16404a') * (1 - dk) + G.hx('#5a3050') * dk
        top, bot = '#%02x%02x%02x' % tuple((top * 255).astype(int)), '#%02x%02x%02x' % tuple((bot * 255).astype(int))
        if kind == 'cyl':
            r = w / 2
            for q in prism_faces(x, z + r, r, 12, 0, h):
                q = np.array(q)
                nrm = np.cross(q[1] - q[0], q[3] - q[0])
                if np.dot(cam.eye - q.mean(0), nrm) <= 0:
                    continue
                wm = np.linalg.norm(q[1] - q[0])
                lightness = 0.55 + 0.45 * max(0, np.dot(nrm / np.linalg.norm(nrm), [0.5, 0, -0.85]))
                tex, e = tower_tex(rs, wm, h, top, bot, near_pal, style='band', lit_p=0.5, ppm=2.2)
                st.face(q, tex * lightness, emit=e, lit=False, outline=None)
            beacons.append((x, h + 2, z + r))
            continue
        levels = [(0, h, 1.0)] if kind == 'box' else [(0, h * 0.6, 1.0), (h * 0.6, h * 0.86, 0.78), (h * 0.86, h, 0.52)]
        for y0, y1, sc in levels:
            ww, dd = w * sc, d * sc
            tex, e = tower_tex(rs, ww, y1 - y0, top, bot, near_pal, rim='#6fdcd0',
                               rim_side='right' if x < 0 else 'left', lit_p=0.45, ppm=2.2,
                               bands=[0.01] if sc < 1 else None, style=['dash', 'band'][rs.integers(2)])
            st.face([(x - ww / 2, y1, z), (x + ww / 2, y1, z), (x + ww / 2, y0, z), (x - ww / 2, y0, z)], tex,
                    emit=e, lit=False, outline=None)
            side, se = tower_tex(rs, dd, y1 - y0, top, bot, near_pal, lit_p=0.35, ppm=2.2)
            if x > 0:
                st.face([(x - ww / 2, y1, z + dd), (x - ww / 2, y1, z), (x - ww / 2, y0, z), (x - ww / 2, y0, z + dd)],
                        side * 0.55, emit=se * 0.55, lit=False, outline=None)
            else:
                st.face([(x + ww / 2, y1, z), (x + ww / 2, y1, z + dd), (x + ww / 2, y0, z + dd), (x + ww / 2, y0, z)],
                        side * 0.55, emit=se * 0.55, lit=False, outline=None)
        beacons += [(x - w / 3, h + 1, z), (x + w / 3, h + 1, z)]
    return cam, st, beacons


def main(seed=8):
    OUT.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(seed)
    bg, bge = backdrop(rs)
    cam, st, beacons = build(rs)
    img, emit = st.render(bg.astype(np.float32), ss=2)
    emit += bge
    for b in beacons:     # red aircraft beacons on every roof
        p, _ = cam.project(np.array(b, float))
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.circle(img, (int(p[0]), int(p[1])), 2, (1.0, 0.2, 0.15), -1, cv2.LINE_AA)
            cv2.circle(emit, (int(p[0]), int(p[1])), 3, (1.0, 0.15, 0.1), -1, cv2.LINE_AA)
    img = wobble(img, rs, amp=0.6)
    out = develop(img, emit, seed=seed, lens=0.8, texbank=True)
    Image.fromarray((out * 255 + .5).astype(np.uint8)).save(OUT / 'o04_skyline.png')
    print('wrote', OUT / 'o04_skyline.png')


if __name__ == '__main__':
    main()
