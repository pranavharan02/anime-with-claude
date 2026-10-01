"""Shots of the original city, set up per refs/akira/CITY_BIBLE.md section D.

    python shots.py                 # all shots
    python shots.py canyon skyline  # some

Output: out/city/<shot>.png
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image

from engine import gouache as G
from engine.citygen import City
from engine.stage import Camera, Stage
from engine.print88 import develop, wobble

OUT = ROOT / 'out' / 'city'
W, H = 1280, 718

# sky: warm near-black, crimson glow, or blue-violet to plum [17]
SKIES = {
    'crimson': ('#120a08', '#7e3136', '#ffb070'),
    'warm': ('#100b06', '#3a2418', '#ffcf90'),
    'violet': ('#1a1a2c', '#606d91', '#d8b0c0'),
}

SHOTS = {
    # one-point canyon: VP on the centre axis, horizon at ~66% of frame [24]
    'canyon': dict(fov=44, sky='crimson', haze='#826f82'),
    # worm's-eye: road level, towers fill the top 80% [24]
    'worm': dict(fov=50, sky='violet', haze='#826f82'),
    # bird's-eye: steep look down a canyon [24]
    'bird': dict(fov=55, sky='warm', haze='#6a5a6a'),
    # skyline across distance as a flat, telephoto [25]
    'skyline': dict(fov=14, sky='crimson', haze='#2a1620'),
    'skyline2': dict(fov=11, sky='warm', haze='#2a1a14'),
    'skyline3': dict(fov=18, sky='violet', haze='#1e1e34'),
    # on the expressway deck, one-point [6, 24]
    'highway': dict(fov=58, sky='violet', haze='#826f82'),
}


def camera_for(city, name):
    ax, cz = city.avenues, city.core[1]
    mid = len(ax) // 2
    if name == 'canyon':
        x = ax[mid + 1]
        eye = (x, 1.3, cz - 900)
        # pitch so the horizon (VP) sits at 66% down the frame
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=eye, yaw=0, pitch=_pitch_for(0.66, SHOTS[name]['fov']))
    if name == 'worm':
        x = ax[mid]
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(x + 2, 0.6, cz - 520), yaw=3, pitch=24)
    if name == 'bird':
        x = (ax[mid] + ax[mid + 1]) / 2
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(ax[mid + 1] - 2, 340, cz - 620), yaw=0, pitch=-52)
    if name == 'skyline2':
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(380, 25, cz - 5200), yaw=-5.0,
                      pitch=_pitch_for(0.6, SHOTS[name]['fov']) + 2.5)
    if name == 'skyline3':
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(-900, 90, cz - 3600), yaw=13.0,
                      pitch=_pitch_for(0.55, SHOTS[name]['fov']) + 4.0)
    if name == 'skyline':
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(-420, 40, cz - 4200), yaw=4.5,
                      pitch=_pitch_for(0.56, SHOTS[name]['fov']) + 3.2)
    if name == 'highway':
        d = city.decks[0]
        return Camera(W, H, fov_deg=SHOTS[name]['fov'], eye=(d['x'] - 3, d['y'] + 1.3, cz - 1300), yaw=0,
                      pitch=_pitch_for(0.64, SHOTS[name]['fov']))
    raise KeyError(name)


def _pitch_for(horizon_frac, fov):
    """Pitch that puts the horizon at horizon_frac of the frame height."""
    f = (W / 2) / np.tan(np.radians(fov) / 2)
    dy = (horizon_frac - 0.5) * H
    return float(np.degrees(np.arctan(dy / f)))


def backdrop(rs, sky, horizon_y):
    top, low, hot = (G.hx(c) for c in SKIES[sky])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(yy / max(horizon_y, 1), 0, 1)[..., None] ** 1.4
    img = top * (1 - t) + low * t
    glow = np.exp(-(((xx - 0.5 * W) / (0.7 * W)) ** 2 + ((yy - horizon_y) / 90) ** 2))[..., None]
    img = img + glow * hot * 0.22
    # airbrush mottle: the spray never lays perfectly even
    img *= (1 + 0.05 * G.field(rs, H, W, 60, 90) + 0.025 * G.field(rs, H, W, 14, 20))[..., None]
    # airbrush on board: soft horizontal streaks and a slight value banding
    img *= (1 + 0.04 * G.field(rs, H, W, 18, 400))[..., None]
    img = np.round(img * 96) / 96 * 0.6 + img * 0.4
    # a couple of thin cloud bands, cel-painted in two flat tones
    for _ in range(0):
        cy = rs.uniform(0.15, 0.6) * horizon_y
        band = np.exp(-((yy - cy - 12 * np.sin(xx / rs.uniform(90, 200) + rs.uniform(0, 6))) / rs.uniform(6, 14)) ** 2)
        band = (band > 0.5).astype(np.float32) * 0.6 + (band > 0.8).astype(np.float32) * 0.4
        img = img * (1 - 0.35 * band[..., None]) + (low * 1.25)[None, None] * 0.35 * band[..., None]
    return img.astype(np.float32)


def searchlights(img, emit, rs, horizon_y, n=5):
    """4-6 straight tapering beams plus a thin coloured line or two [20]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    for i in range(n + 2):
        x0 = rs.uniform(0.15, 0.85) * W
        y0 = horizon_y + rs.uniform(-40, 20)
        ang = np.radians(rs.uniform(-35, 35))
        thin = False
        if i >= n:
            continue
        width0, width1 = 6.0, rs.uniform(40, 70)
        col = G.hx(['#9ff0ff', '#ff70d0'][i % 2]) if thin else G.hx('#f2f4ff')
        L = 1400
        x1, y1 = x0 + np.sin(ang) * L, y0 - np.cos(ang) * L
        d = np.array([x1 - x0, y1 - y0]) / L
        px, py = xx - x0, yy - y0
        along = px * d[0] + py * d[1]
        across = np.abs(px * -d[1] + py * d[0])
        wdt = width0 + (width1 - width0) * np.clip(along / L, 0, 1)
        fall = (along > 0) * np.clip(1 - along / L, 0, 1) ** 0.7
        mott = 1 + 0.25 * G.field(rs, H, W, 40, 40)
        for pas, (wf, a) in enumerate(((1.0, 0.07), (0.6, 0.08), (0.3, 0.1), (0.1, 0.12))):   # airbrush build-up
            m = np.clip(1 - across / (wdt * wf), 0, 1) ** 1.5 * fall * mott
            img += m[..., None] * col * a
            emit += m[..., None] * col * a * 0.25


def _seen(cam, zbuf, p, d):
    x, y = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
    return d <= zbuf[y, x] + 3.0


def lamps(img, emit, cam, city, zbuf):
    """Lamp = solid core + airbrushed halo 3-5x its radius; flat pools below [19]."""
    for lx, ly, lz in city.lamps:
        p, d = cam.project(np.array([lx, ly, lz], float))
        if d < 2 or d > 1800 or not (0 <= p[0] < W and 0 <= p[1] < H) or not _seen(cam, zbuf, p, d):
            continue
        r = max(0.8, 160 / d)
        cv2.circle(img, (int(p[0] * 4), int(p[1] * 4)), int(r * 4), (1.0, 0.92, 0.75), -1, cv2.LINE_AA, shift=2)
        cv2.circle(emit, (int(p[0] * 4), int(p[1] * 4)), int(r * 4 * 1.2), (1.0, 0.75, 0.4), -1, cv2.LINE_AA, shift=2)
        if d < 400 and ly < 12:                      # the pool on the road under it
            ring = np.array([cam.project(np.array([lx + 6 * np.cos(a), 0.02, lz + 6 * np.sin(a)]))[0]
                             for a in np.linspace(0, 2 * np.pi, 20, endpoint=False)], np.float32)
            pool = np.zeros((H, W), np.float32)
            cv2.fillPoly(pool, [np.round(ring).astype(np.int32)], 1.0, cv2.LINE_AA)
            pool = cv2.GaussianBlur(pool, (0, 0), float(np.clip(120 / d, 1.5, 12)))
            img += pool[..., None] * G.hx('#ffb060') * 0.10
    halo = cv2.GaussianBlur(emit, (0, 0), 4) * 0.8
    emit += halo


def beacons(img, emit, cam, city, zbuf):
    for b in city.beacons:
        p, d = cam.project(np.array(b, float))
        if d > 2 and 0 <= p[0] < W and 0 <= p[1] < H and _seen(cam, zbuf, p, d):
            r = max(0.8, 220 / d)
            cv2.circle(img, (int(p[0] * 4), int(p[1] * 4)), int(r * 4), (1.0, 0.18, 0.12), -1, cv2.LINE_AA, shift=2)
            cv2.circle(emit, (int(p[0] * 4), int(p[1] * 4)), int(r * 6), (1.0, 0.12, 0.08), -1, cv2.LINE_AA, shift=2)


CAR_COLS = ['#c8c8c0', '#5a6270', '#2a3448', '#6a2830', '#8a8a6a', '#3a4a3a', '#d8d0b8']


def cel_cars(st, cam, city, rs, n=40, max_d=320):
    """Traffic as inked cels: flat two-tone boxes with ink outlines [B, cels]."""
    eye = cam.eye
    placed = 0
    tries = 0
    while placed < n and tries < 2000:
        tries += 1
        x = city.avenues[rs.integers(len(city.avenues))]
        w = city.aw[x]
        lane = rs.choice([-0.3, -0.12, 0.12, 0.3]) * w
        z = eye[2] + rs.uniform(8, max_d)
        cx = x + lane
        if np.hypot(cx - eye[0], z - eye[2]) > max_d:
            continue
        ln, wd, hb, hc = rs.uniform(4.0, 4.8), 1.75, 0.75, 0.6
        col = G.hx(CAR_COLS[rs.integers(len(CAR_COLS))])
        z0, z1 = z, z + ln
        x0, x1 = cx - wd / 2, cx + wd / 2
        def face(q, c, emit=None):
            t = np.empty((8, 8, 3), np.float32)
            t[:] = c
            st.face(q, t, emit=emit, lit=False, outline='#140c12', line_w=1.0)
        lit, sh = col * 0.9, col * 0.5
        y0, y1 = 0.25, 0.25 + hb
        face([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], sh)                       # tail
        if eye[0] < x0:
            face([(x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1)], lit)
        elif eye[0] > x1:
            face([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], lit)
        face([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], col)                       # bonnet / boot
        cz0, cz1 = z0 + ln * 0.25, z0 + ln * 0.7
        face([(x0 + 0.15, y1 + hc, cz0 + 0.3), (x1 - 0.15, y1 + hc, cz0 + 0.3), (x1 - 0.1, y1, cz0), (x0 + 0.1, y1, cz0)],
             G.hx('#2a3040'))                                                                       # rear glass
        face([(x0 + 0.15, y1 + hc, cz1), (x1 - 0.15, y1 + hc, cz1), (x1 - 0.15, y1 + hc, cz0 + 0.3),
              (x0 + 0.15, y1 + hc, cz0 + 0.3)], col * 0.8)                                         # roof
        tail = np.empty((4, 8, 3), np.float32)
        tail[:] = G.hx('#ff3a28')
        for tx in (x0 + 0.1, x1 - 0.45):
            st.face([(tx, 0.85, z0 - 0.01), (tx + 0.35, 0.85, z0 - 0.01), (tx + 0.35, 0.7, z0 - 0.01),
                     (tx, 0.7, z0 - 0.01)], tail, emit=tail * 1.2, lit=False)
        placed += 1


def cables(img, cam, rs, n=4):
    """Overhead wires sagging across the street, a foreground occluder [9, 27]."""
    for _ in range(n):
        x0, x1 = cam.eye[0] - rs.uniform(8, 14), cam.eye[0] + rs.uniform(8, 14)
        z = cam.eye[2] + rs.uniform(6, 30)
        y = rs.uniform(7, 10)
        sag, dz = rs.uniform(0.9, 1.6), rs.uniform(-1, 3)
        pts = []
        for t in np.linspace(0, 1, 40):
            P = np.array([x0 + (x1 - x0) * t, y - sag * 4 * t * (1 - t), z + dz * t])
            q, d = cam.project(P)
            pts.append(q)
        pts = np.array(pts)
        for i in range(len(pts) - 1):   # inked: the pen thickens and thins along the wire
            wdt = 1 + int(2.5 * (0.5 + 0.5 * np.sin(i / 5 + rs.uniform(0, 6))))
            cv2.line(img, tuple(np.round(pts[i] * 4).astype(int)), tuple(np.round(pts[i + 1] * 4).astype(int)),
                     (0.05, 0.04, 0.06), wdt, cv2.LINE_AA, shift=2)


BILL_WORDS = ['CRAFT', 'NEURON', 'KIRIN-DO', 'OASIS', 'VOLT 88', 'MIRAGE', 'TOKAI', 'SPARK']


def billboards(st, cam, city, rs, n=12, max_d=1400):
    """Painted billboard frames on roofs, with invented brand words [7, 8]."""
    eye = cam.eye
    cands = [p for p in city.parts if p['kind'] == 'box' and 25 < p['b'][3] < 180]
    rs.shuffle(cands)
    k = 0
    for pt in cands:
        x0, x1, y0, y1, z0, z1 = pt['b']
        cz_ = z0 - 0.5
        d = np.linalg.norm(np.array([(x0 + x1) / 2, y1, cz_]) - eye)
        if d > max_d or eye[2] > z0 or x1 - x0 < 14:
            continue
        bw, bh = min(16, x1 - x0 - 2), rs.uniform(4, 7)
        bx = (x0 + x1) / 2 - bw / 2
        q = [(bx, y1 + 2 + bh, cz_), (bx + bw, y1 + 2 + bh, cz_), (bx + bw, y1 + 2, cz_), (bx, y1 + 2, cz_)]
        bg = ['#e8e0c8', '#d8302a', '#2a5a9a', '#f0c030', '#1e1e24'][rs.integers(5)]
        fg = '#1e1e24' if bg in ('#e8e0c8', '#f0c030') else '#f4f0e0'
        t = G.sign(int(bw * 20), int(bh * 20), BILL_WORDS[rs.integers(len(BILL_WORDS))], fg=fg, bg=bg,
                   border='#2a2a30', vertical=False, font='C:/Windows/Fonts/arialbd.ttf')
        st.face(q, t * 0.85, emit=t * 0.25, lit=False)
        k += 1
        if k >= n:
            break


def soften_by_depth(img, zbuf):
    """Far stock dissolves: blur grows with distance, like painted haze [21]."""
    far = np.clip((np.log(np.maximum(zbuf, 1)) - np.log(500)) / np.log(8), 0, 1)[..., None]
    soft = cv2.GaussianBlur(img, (0, 0), 1.3)
    return img * (1 - far) + soft * far


def shoot(city, name, seed=None, quality=0.9):
    seed = (abs(hash(name)) % 1000) if seed is None else seed
    t0 = time.time()
    rs = np.random.default_rng(seed)
    sh = SHOTS[name]
    cam = camera_for(city, name)
    hz, _ = cam.project(np.array([cam.eye[0], cam.eye[1], cam.eye[2] + 20000.0]))
    hy = float(np.clip(hz[1], 30, H - 10))
    st = Stage(cam, ambient=(0.16, 0.18, 0.26))
    for lx, ly, lz in city.lamps:
        if np.linalg.norm(np.array([lx, ly, lz]) - cam.eye) < 200:
            st.light((lx, ly, lz), (1.0, 0.68, 0.34), 1.4, 7.0)
    city.populate(st, cam, ss=2, quality=quality, seed=seed, haze=(sh['haze'], (250, 600, 1200, 2200)))
    billboards(st, cam, city, rs)
    if name in ('canyon', 'highway', 'worm', 'bird'):
        cel_cars(st, cam, city, rs, n=60 if name == 'bird' else 30, max_d=600 if name == 'bird' else 260)
    bg = backdrop(rs, sh['sky'], hy)
    bg_emit = np.zeros_like(bg)
    if name.startswith('skyline'):          # beams rise from behind the city, so paint them on the backdrop
        searchlights(bg, bg_emit, np.random.default_rng(abs(hash(name)) % 10000), hy + 40)
    img, emit = st.render(bg, ss=2)
    emit += bg_emit
    if not name.startswith('skyline'):
        img = soften_by_depth(img, st.zbuf)
    if name in ('canyon', 'worm'):
        cables(img, cam, rs)
    lamps(img, emit, cam, city, st.zbuf)
    beacons(img, emit, cam, city, st.zbuf)
    img = wobble(img, rs, amp=0.4)
    img = cv2.GaussianBlur(img, (0, 0), 0.3 if name.startswith('skyline') else 0.45)   # chalky edges [16]
    out = develop(img, emit, seed=seed, lens=0.6, texbank=True)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255 + .5).astype(np.uint8)).save(OUT / f'{name}.png')
    print(f'{name}: {len(st.faces)} faces, {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    city = City(seed=7)
    print(len(city.parts), 'parts,', len(city.beacons), 'beacons', flush=True)
    for n in (sys.argv[1:] or list(SHOTS)):
        shoot(city, n)
