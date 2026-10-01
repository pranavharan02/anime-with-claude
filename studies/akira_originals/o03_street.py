"""Original 03: "Street". A night street in perspective, painted surface by surface.

Built on engine/stage.py (3D quads, point lights, homography warp) and
engine/gouache.py (poster-colour texture painting). The cel layer (bike,
figure, cables) is vector on top. Shot through engine/print88.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import skia
from PIL import Image

from engine import gouache as G
from engine.stage import Camera, Stage, quad_x, quad_y, quad_z
from engine.print88 import develop, wobble
from engine.paintover import paint_over
from engine.ink import curve

sys.path.insert(0, str(ROOT / 'studies' / 'akira'))
import bike as BIKE

OUT = ROOT / 'out' / 'akira_originals'
W, H = 1280, 718
PPM = 70  # texture pixels per metre
PAINT_OVER = False


def tex_for(wm, hm, ppm=PPM):
    return int(max(8, wm * ppm)), int(max(8, hm * ppm))


# ------------------------------------------------------------------ backdrop --
def backdrop(rs):
    """Sky and the far city in violet haze: painted flat, behind everything."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip(yy / 440, 0, 1)[..., None]
    sky = G.hx('#02060d') * (1 - t) + G.hx('#142a3a') * t
    glow = np.exp(-(((xx - 930) / 420) ** 2 + ((yy - 430) / 140) ** 2))[..., None]
    img = sky + glow * G.hx('#6a4e7e') * 0.35
    img = img * (1 + 0.035 * G.field(rs, H, W, 40, 160)[..., None] + 0.02 * G.field(rs, H, W, 8, 60)[..., None])
    emit = np.zeros_like(img)
    # far towers: layered silhouettes, lighter with distance, soft lit windows
    for layer, (col, top, base, wmin, wmax) in enumerate((
            ('#4f6a80', 250, 445, 18, 50), ('#3a4e66', 210, 450, 22, 60), ('#28364c', 170, 455, 26, 70))):
        x = 560 + rs.uniform(-20, 0)
        while x < W + 40:
            w = rs.uniform(wmin, wmax)
            tp = rs.uniform(top, base - 40)
            c = G.hx(col) * rs.uniform(0.92, 1.08)
            m = G.rough_mask(H, W, [[(x, tp), (x + w, tp), (x + w, base), (x, base)]], rs, 0.3)
            G.blend(img, m, c)
            # windows: tiny warm and pink dabs
            for _ in range(int(w * (base - tp) / 260)):
                wx, wy = rs.uniform(x + 2, x + w - 3), rs.uniform(tp + 3, base - 4)
                k = ['#f0d9a0', '#f2b6d6', '#cfe2ff', '#ffd27a'][rs.integers(4)]
                cv2.rectangle(img, (int(wx), int(wy)), (int(wx + 2), int(wy + 1)), tuple(float(v) for v in G.hx(k) * 0.7), -1)
                cv2.rectangle(emit, (int(wx), int(wy)), (int(wx + 2), int(wy + 1)), tuple(float(v) for v in G.hx(k) * 0.12), -1)
            x += w * rs.uniform(0.7, 1.05)
        # haze over each layer
        hz = np.clip((yy - (base - 120)) / 140, 0, 1)[..., None] * 0.35
        img = img * (1 - hz) + G.hx('#5e7a8e') * hz
    return img, emit


# ------------------------------------------------------------------- facades --
def facade_b1(rs, wm, hm):
    """Concrete, pale blue-grey: a shop shutter below, four floors of windows."""
    tw, th = tex_for(wm, hm)
    t = G.wall(th, tw, '#8792a3', rs, var=0.05)
    e = np.zeros_like(t)
    fl = th / hm  # px per metre vertically
    G.shutter(t, rs, tw * 0.06, th - 3.0 * fl, tw * 0.62, th - 0.05 * fl, '#7d8494')
    G.dab(t, rs, tw * 0.66, th - 3.0 * fl, tw * 0.95, th - 0.05 * fl, '#2b3140')           # shop door recess
    G.dab(t, rs, tw * 0.70, th - 2.6 * fl, tw * 0.91, th - 0.6 * fl, '#d8e6c0')            # lit glass door
    G.blend(e, G.rough_mask(th, tw, [[(tw * .70, th - 2.6 * fl), (tw * .91, th - 2.6 * fl),
                                        (tw * .91, th - .6 * fl), (tw * .70, th - .6 * fl)]], rs), '#c8e0b0', 0.25)
    G.ruler(t, (0, th - 3.2 * fl), (tw, th - 3.2 * fl), '#5c6474', 4)
    G.windows(t, e, rs, tw * 0.05, th - 15.5 * fl, tw * 0.95, th - 3.6 * fl, cols=6, rows=4, lit_p=0.3,
              frame='#4a5262', glass='#26304a')
    G.drips(t, rs, 40, '#6a4a40', y_from=[int(th - 15.5 * fl + k * (11.9 * fl / 4)) for k in range(1, 5)],
            length=(20, 140), alpha=(0.15, 0.45))
    G.stains(t, rs, 14, '#4c5466', size=(30, 120))
    G.cracks(t, rs, 6, '#4a505e')
    G.ruler(t, (0, 2), (tw, 2), '#c3cad6', 3, 0.8)  # lit parapet edge
    return t, e


def facade_b2(rs, wm, hm):
    """Cream tile, older: dense windows, heavy rust runs."""
    tw, th = tex_for(wm, hm)
    t = G.wall(th, tw, '#b0a78e', rs, var=0.05)
    e = np.zeros_like(t)
    fl = th / hm
    G.dab(t, rs, 0, th - 3.4 * fl, tw, th, '#5a5048')
    G.dab(t, rs, tw * 0.1, th - 2.8 * fl, tw * 0.45, th - 0.1 * fl, '#f0d79a')               # lit bar front
    G.blend(e, G.rough_mask(th, tw, [[(tw * .1, th - 2.8 * fl), (tw * .45, th - 2.8 * fl),
                                        (tw * .45, th - .1 * fl), (tw * .1, th - .1 * fl)]], rs), '#f0c070', 0.25)
    G.dab(t, rs, tw * 0.13, th - 2.5 * fl, tw * 0.42, th - 1.2 * fl, '#a8724a')              # noren curtain
    G.windows(t, e, rs, tw * 0.04, th - 21.5 * fl, tw * 0.96, th - 3.8 * fl, cols=7, rows=6, lit_p=0.45,
              frame='#6e6656', glass='#2a3040')
    for y in np.linspace(th - 21.5 * fl, th - 3.8 * fl, 7):
        G.ruler(t, (0, y), (tw, y), '#8c836c', 3, 0.7)
    G.drips(t, rs, 90, '#7a4a32', y_from=[int(y) for y in np.linspace(th - 21.5 * fl, th - 3.8 * fl, 7)],
            length=(30, 200), width=(1, 5), alpha=(0.2, 0.6))
    G.stains(t, rs, 20, '#6f6656', size=(30, 140))
    return t, e


def facade_b3(rs, wm, hm):
    """Dark brick, low: few lights, a fire escape ladder."""
    tw, th = tex_for(wm, hm)
    t = G.wall(th, tw, '#5e4650', rs, var=0.06)
    e = np.zeros_like(t)
    fl = th / hm
    for y in np.arange(4, th, 9):           # brick courses, barely there
        G.ruler(t, (0, y), (tw, y), '#4e3a44', 1, 0.35)
    G.windows(t, e, rs, tw * 0.08, th - 11 * fl, tw * 0.92, th - 3.5 * fl, cols=5, rows=3, lit_p=0.25,
              frame='#3a2a32', glass='#1c1e2c')
    G.shutter(t, rs, tw * 0.1, th - 3.0 * fl, tw * 0.9, th - 0.05 * fl, '#5a5866')
    x = tw * 0.55
    G.ruler(t, (x, th - 11 * fl), (x, th - 3 * fl), '#2a2028', 3)
    G.ruler(t, (x + 20, th - 11 * fl), (x + 20, th - 3 * fl), '#2a2028', 3)
    for y in np.arange(th - 11 * fl, th - 3 * fl, 14):
        G.ruler(t, (x, y), (x + 20, y), '#2a2028', 2)
    G.drips(t, rs, 30, '#3a2a2a', alpha=(0.2, 0.5))
    return t, e


def facade_b4(rs, wm, hm):
    """A tall dark office block further down the street."""
    tw, th = tex_for(wm, hm, ppm=40)
    t = G.wall(th, tw, '#3c4762', rs, var=0.05)
    e = np.zeros_like(t)
    G.windows(t, e, rs, tw * 0.03, th * 0.03, tw * 0.97, th * 0.9, cols=22, rows=18, lit_p=0.26,
              frame='#2c3448', glass='#1a2236', gap=0.35,
              lit=('#f2e2a0', '#e8cf86'))
    return t, e


def side_wall(rs, wm, hm, color):
    tw, th = tex_for(wm, hm, ppm=40)
    t = G.wall(th, tw, color, rs, var=0.05)
    G.drips(t, rs, 20, '#2a2a34', alpha=(0.1, 0.35))
    G.ruler(t, (tw - 2, 0), (tw - 2, th), '#9aa0b4', 2, 0.6)   # rim of the corner, catching sky
    return t


def ground(rs, wm, dm, kind):
    tw, th = tex_for(wm, dm, ppm=24)
    if kind == 'road':
        t = G.wall(th, tw, '#2c3446', rs, streak='h', var=0.06, dark_bottom=0)
        for x in (tw * 0.42,):
            for y in np.arange(0, th, 90):
                G.dab(t, rs, x - 3, y, x + 3, y + 45, '#8a8e9a', alpha=0.8, rough=0.8)
        G.stains(t, rs, 30, '#1e2432', size=(20, 90))
        G.cracks(t, rs, 20, '#1a1e2a', length=(20, 80))
    else:
        t = G.wall(th, tw, '#5c6476', rs, streak='h', var=0.05, dark_bottom=0)
        for y in np.arange(0, th, 32):          # paving joints
            G.ruler(t, (0, y), (tw, y), '#444b5a', 1.2, 0.7)
        G.stains(t, rs, 30, '#454c5c', size=(10, 50))
        G.cracks(t, rs, 30, '#3a404e', length=(10, 50))
        for _ in range(30):                   # litter: paper, cans
            x, y = rs.uniform(0, tw), rs.uniform(0, th)
            k = ['#d8d6cc', '#b04040', '#9aa0aa', '#c8b080'][rs.integers(4)]
            G.dab(t, rs, x, y, x + rs.uniform(2, 6), y + rs.uniform(1, 4), k, rough=0.9)
    return t


def vending(rs):
    tw, th = 80, 150
    t = G.wall(th, tw, '#c8d4e6', rs, var=0.03, dark_bottom=0)
    e = np.zeros_like(t)
    G.dab(t, rs, 6, 8, 74, 70, '#eef6ff')
    for r in range(3):
        for c in range(5):
            k = ['#d84040', '#3a7ad8', '#e8c040', '#40a060', '#f0f0f0'][(r * 2 + c) % 5]
            G.dab(t, rs, 10 + c * 12, 14 + r * 18, 17 + c * 12, 26 + r * 18, k, rough=0.4)
    G.dab(t, rs, 10, 80, 50, 100, '#2a3040')
    G.dab(t, rs, 56, 82, 70, 120, '#9aa6b8')
    G.dab(t, rs, 8, 126, 72, 142, '#1c2230')
    e[:] = t * 0.28
    return t, e


def sign_face(text, rs, bg, fg, w=60, h=220, lit=0.6):
    t = G.sign(w, h, text, fg=fg, bg=bg)
    t = t * (1 + 0.03 * G.field(rs, h, w, 20, 10)[..., None])
    return t, t * lit


# ------------------------------------------------------------------ the shot --
def build(rs):
    cam = Camera(W, H, fov_deg=58, eye=(3.0, 1.45, 0), yaw=-14, pitch=4)
    st = Stage(cam, ambient=(0.19, 0.24, 0.36))
    st.light((7.0, 6.0, 26), (1.0, 0.68, 0.32), 2.8, 4.5)          # sodium streetlamp
    st.light((-3.0, 1.3, 11.0), (0.75, 0.9, 1.0), 0.7, 2.0)         # vending machine
    st.light((-2.6, 2.0, 20.5), (1.0, 0.75, 0.45), 0.6, 2.2)        # the bar's open front
    st.light((-3.2, 5.0, 17.0), (1.0, 0.45, 0.7), 0.6, 2.5)         # neon sign

    X = -4.2
    blds = [(7, 14, 16, facade_b1), (14, 25, 22, facade_b2), (25, 37, 12, facade_b3), (37, 75, 30, facade_b4)]
    for z0, z1, hgt, fn in blds:
        t, e = fn(rs, z1 - z0, hgt)
        st.face(quad_x(X, z0, z1, 0, hgt), t, emit=e)
    # step faces where a taller building rises behind a shorter one
    st.face(quad_z(14, -16, X, 16, 22), side_wall(rs, 12, 6, '#8e8673'))
    st.face(quad_z(37, -16, X, 12, 30), side_wall(rs, 12, 18, '#2c3448'))
    # parapets / roof edges catching sky light
    for z0, z1, hgt, _ in blds[:3]:
        t = G.wall(10, int((z1 - z0) * 40), '#a4acc0', rs, var=0.04, dark_bottom=0)
        st.face(quad_y(hgt + 0.02, -6, X, z0, z1), t * 0.6, lit=False)

    # right side: darker buildings, set back, a lit sign or two
    rb = [(18, 30, 14, '#3a3e54'), (30, 48, 24, '#2e3448'), (48, 80, 18, '#353a50')]
    for z0, z1, hgt, col in rb:
        tw, th = tex_for(z1 - z0, hgt, 40)
        t = G.wall(th, tw, col, rs, var=0.05)
        e = np.zeros_like(t)
        G.windows(t, e, rs, tw * 0.05, th * 0.06, tw * 0.95, th * 0.85, cols=int((z1 - z0) / 1.6),
                  rows=int(hgt / 3.2), lit_p=0.22, frame='#262a3a', glass='#161c2c',
                  lit=('#f0dc98', '#e4c47e'))
        st.face(quad_x(11.5, z1, z0, 0, hgt), t, emit=e)

    # ground: road, left pavement and its kerb, right pavement
    st.face(quad_y(0.0, -1.6, 11.5, 3, 140), ground(rs, 13.1, 137, 'road'))
    st.face(quad_y(0.15, X, -1.6, 3, 140), ground(rs, 2.6, 137, 'pave'))
    st.face(quad_x(-1.6, 3, 140, 0, 0.15), G.wall(8, 400, '#8a90a0', rs, var=0.05), order=-0.1)
    st.face(quad_y(0.15, 9.5, 11.5, 14, 140), ground(rs, 2.0, 126, 'pave'))

    # overpass crossing the street, with piers
    tw, th = tex_for(70, 1.4, 20)
    t = G.wall(th, tw, '#4a4c62', rs, streak='h', var=0.06, dark_bottom=0)
    G.ruler(t, (0, 2), (tw, 2), '#9a9ab4', 2)
    G.drips(t, rs, 80, '#2a2a3a', alpha=(0.2, 0.5), length=(5, 20))
    st.face(quad_z(55, -30, 40, 7.0, 8.4), t)
    st.face(quad_y(7.0, -30, 40, 55, 62), G.wall(40, 300, '#1e2030', rs), lit=False)
    for px in (-2.5, 9.0):
        t = G.wall(140, 24, '#3e4258', rs, var=0.06)
        G.drips(t, rs, 6, '#2a2a3a')
        st.face(quad_z(56, px, px + 1.3, 0, 7.0), t)

    # vending machine against B1, and its side
    t, e = vending(rs)
    st.face(quad_x(-3.45, 10.0, 11.0, 0.15, 2.0), t, emit=e)
    st.face(quad_z(10.0, X, -3.45, 0.15, 2.0), G.wall(60, 30, '#9aa6ba', rs) * 0.8)

    # vertical signboards sticking out from the facades
    for z, y0, y1, text, bg, fg in ((12.5, 5.5, 10.5, 'スナック', '#efe6cf', '#c0283c'),
                                     (17.0, 4.0, 9.5, '麻雀荘', '#f2a8c8', '#2a1830'),
                                     (21.5, 3.6, 7.0, '喫茶', '#e8e0b0', '#2a3060'),
                                     (28.0, 4.0, 8.5, '質', '#d8dce6', '#1c2030')):
        t, e = sign_face(text, rs, bg, fg)
        st.face(quad_z(z, X, X + 0.85, y0, y1), t, emit=e, lit=True)

    clutter(st, rs, X)

    # the streetlamp: pole, arm and head
    pole = G.wall(400, 12, '#b8a24c', rs, var=0.05)
    st.face(quad_z(26.0, 8.6, 8.78, 0.15, 6.3), pole)
    st.face([(8.7, 6.35, 26), (6.6, 6.2, 26), (6.6, 6.08, 26), (8.7, 6.2, 26)], G.wall(6, 120, '#b8a24c', rs))
    head = np.ones((16, 50, 3), np.float32) * G.hx('#ffd890')
    st.face([(6.5, 6.14, 26), (7.5, 6.14, 26), (7.5, 5.98, 26), (6.5, 5.98, 26)], head, emit=head * 1.2, lit=False)
    return cam, st


def clutter(st, rs, X):
    """The stuff Mizutani never left out: units, pipes, posters, bins, a pole."""
    def tx(w, h, col, var=0.06):
        t = G.wall(h, w, col, rs, var=var)
        G.drips(t, rs, max(1, w // 12), '#3a2e2e', alpha=(0.15, 0.4), length=(4, h * 0.8))
        return t
    # air-conditioner units hung on the facades
    for z, y in ((8.5, 4.2), (11.2, 7.6), (16.0, 6.2), (19.5, 10.4), (23.0, 4.8), (27.0, 6.5)):
        f = tx(60, 44, '#b8bcc4')
        for gy in range(8, 40, 5):
            G.ruler(f, (6, gy), (36, gy), '#6a6e78', 1.2)
        G.dab(rs=rs, tex=f, x0=40, y0=8, x1=56, y1=38, color='#8a8e98')
        st.box(X, X + 0.55, y, y + 0.55, z, z + 0.8, f, tex_side=tx(40, 44, '#9a9ea8'), tex_top=tx(60, 40, '#c8ccd4'))
    # drainpipes and conduits down the walls
    for z in (13.9, 24.9, 30.5):
        st.face(quad_x(X + 0.12, z, z + 0.18, 0, 14), tx(8, 300, '#6a6e7c'))
    # posters pasted at street level, peeling
    for z, y, col in ((7.6, 1.0, '#d8c8a0'), (14.4, 1.2, '#c86a5a'), (15.3, 1.1, '#e8e2d0'), (25.5, 1.0, '#6a8ab0')):
        f = G.wall(60, 44, col, rs, var=0.08)
        for k in range(3):
            G.dab(f, rs, rs.uniform(2, 20), rs.uniform(4, 50), rs.uniform(24, 42), rs.uniform(10, 58),
                  ['#2a2430', '#a03030', '#f4f0e0'][rs.integers(3)], rough=0.8)
        st.face(quad_x(X + 0.02, z, z + 0.6, y, y + 0.85), f, line_w=0.8)
    # bins and a crate on the pavement
    for z, col in ((12.6, '#4a6a5a'), (13.3, '#5a6a7a'), (19.8, '#7a6a4a')):
        st.box(-3.9, -3.3, 0.15, 1.0, z, z + 0.6, tx(30, 40, col), tex_side=tx(30, 40, col) * 0.75,
               tex_top=tx(30, 30, '#2a2a30'))
    # a utility pole on the left kerb with a transformer drum and a lamp bracket
    st.face(quad_z(18.0, -2.0, -1.72, 0.15, 11.5), tx(10, 400, '#7a7468'))
    st.box(-2.25, -1.55, 7.6, 8.6, 17.8, 18.4, tx(30, 40, '#8a8e96'), tex_side=tx(20, 40, '#6a6e76'))
    for y in (9.5, 10.6):
        st.face(quad_z(17.9, -3.2, -0.6, y, y + 0.12), tx(120, 6, '#5a5650'))


def light_cone(img, emit, cam):
    """The lamp's light hanging in the damp air: airbrushed, additive."""
    p, _ = cam.project(np.array([7.0, 6.0, 26.0]))
    g, _ = cam.project(np.array([7.0, 0.0, 26.0]))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    t = np.clip((yy - p[1]) / max(1, g[1] - p[1]), 0, 1)
    half = 18 + 120 * t
    cone = np.clip(1 - np.abs(xx - (p[0] + (g[0] - p[0]) * t)) / half, 0, 1) ** 1.5 * (yy > p[1]) * (1 - 0.5 * t)
    img += cone[..., None] * G.hx('#ffb060') * 0.10
    halo = np.exp(-(((xx - p[0]) / 60) ** 2 + ((yy - p[1]) / 40) ** 2))
    emit += halo[..., None] * G.hx('#ffc070') * 0.35


def cels(cam, rs):
    """Vector cel layer: the parked bike, a figure, overhead cables."""
    info = skia.ImageInfo.Make(W * 2, H * 2, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
    s = skia.Surface.MakeRaster(info)
    c = s.getCanvas()
    c.scale(2, 2)
    INK = skia.Color(26, 8, 14)

    def fill(pts, col):
        poly = curve(pts, closed=True)
        p = skia.Path()
        p.addPoly([skia.Point(float(x), float(y)) for x, y in poly], True)
        c.drawPath(p, skia.Paint(AntiAlias=True, Color=skia.Color(*[int(v * 255) for v in G.hx(col)])))
        return poly

    def ink(pts, w=1.3, closed=False):
        poly = curve(pts, closed=closed)
        p = skia.Path()
        p.addPoly([skia.Point(float(x), float(y)) for x, y in poly], closed)
        c.drawPath(p, skia.Paint(AntiAlias=True, Color=INK, Style=skia.Paint.kStroke_Style, StrokeWidth=w,
                                 StrokeJoin=skia.Paint.kRound_Join, StrokeCap=skia.Paint.kRound_Cap))

    # a lone figure walking under the overpass, small and dark against the glow
    (fx, fy), _ = cam.project(np.array([4.5, 0.0, 50.0]))
    k = cam.f / cam.to_cam(np.array([4.5, 0.0, 50.0]))[2]
    u = lambda a, b: (fx + a * k, fy - b * k)
    body = [u(-0.22, 0.0), u(-0.12, 0.85), u(-0.24, 1.15), u(-0.2, 1.45), u(0.0, 1.55), u(0.2, 1.45),
            u(0.24, 1.15), u(0.1, 0.85), u(0.22, 0.0), u(0.05, 0.0), u(0.0, 0.7), u(-0.05, 0.0)]
    fill(body, '#10131c')
    fill([(fx + 0.11 * k * np.cos(a), fy - 1.68 * k + 0.11 * k * np.sin(a)) for a in np.linspace(0, 6.28, 10, endpoint=False)],
         '#10131c')
    # cables strung across the street overhead
    for (a, b, sag) in (((-4.2, 9.5, 9), (11.5, 8.5, 26), 1.2), ((-4.2, 8.0, 13), (11.5, 7.6, 30), 1.5),
                        ((-4.2, 11.0, 20), (11.5, 9.0, 40), 1.0)):
        A, B = np.array(a), np.array(b)
        pts = []
        for t in np.linspace(0, 1, 30):
            P = A + (B - A) * t
            P[1] -= sag * 4 * t * (1 - t)
            q, z = cam.project(P)
            pts.append(tuple(q))
        ink(pts, 1.4)
    arr = s.makeImageSnapshot().toarray().astype(np.float32) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
    return arr


def road_reflections(img, emit, cam, st):
    """Wet asphalt: render the street mirrored below the ground plane, then
    smear and break it into vertical brush strokes where the ground is wet."""
    rs = np.random.default_rng(11)
    mirror = Stage(cam, ambient=tuple(st.ambient))
    mirror.lights = st.lights
    for fc in st.faces:
        c = fc['c'].copy()
        if np.allclose(c[:, 1], c[0, 1]) and c[0, 1] < 0.5:   # skip the ground itself
            continue
        c[:, 1] = -c[:, 1]
        mirror.face(c, fc['tex'], emit=fc['emit'], alpha=fc['alpha'], lit=fc['lit'], order=fc['order'])
    refl, remit = mirror.render(np.zeros((H, W, 3), np.float32), ss=1)
    refl = refl + remit * 1.2
    m = np.zeros((H, W), np.float32)
    ground = np.array([cam.project(np.array(p, float))[0] for p in
                       [(-4.2, 0, 140), (11.5, 0, 140), (11.5, 0, 4), (-4.2, 0, 4)]], np.float32)
    cv2.fillPoly(m, [np.round(ground).astype(np.int32)], 1.0, cv2.LINE_AA)
    k = np.zeros((41, 1), np.float32)
    k[:, 0] = 1.0 / 41
    refl = cv2.filter2D(refl, -1, k)                               # smeared down the wet surface
    strokes = cv2.resize(rs.random((H // 6, W // 3)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    strokes = np.clip((strokes - 0.35) * 2.2, 0, 1)                 # broken into vertical strokes
    wet = m * (0.35 + 0.45 * strokes)
    img[:] = img * (1 - wet[..., None] * 0.5) + refl * wet[..., None] * 0.55


def trail_and_bike(cam, rs):
    """A capsule bike far down the street, its tail light drawn out behind it
    as a ribbon that hangs in the air back toward the camera."""
    info = skia.ImageInfo.Make(W * 2, H * 2, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
    s = skia.Surface.MakeRaster(info)
    e = skia.Surface.MakeRaster(info)
    c, ce = s.getCanvas(), e.getCanvas()
    for cv in (c, ce):
        cv.scale(2, 2)
    zs = np.linspace(31.0, 5.0, 120)
    xs = 3.4 + 0.9 * np.sin((31 - zs) / 4.2) * np.clip((31 - zs) / 6, 0, 1)
    pts, ws = [], []
    for x, z in zip(xs, zs):
        q, d = cam.project(np.array([x, 0.72, z]))
        pts.append(q)
        ws.append(cam.f * 0.16 / d)
    pts, ws = np.array(pts), np.array(ws)
    tn = np.gradient(pts, axis=0)
    tn /= np.linalg.norm(tn, axis=1, keepdims=True) + 1e-9
    nm = np.stack([-tn[:, 1], tn[:, 0]], 1)
    fade = np.linspace(1, 0.25, len(pts))
    for wf, col, a in ((2.6, (255, 40, 30), 0.25), (1.0, (255, 60, 30), 0.9), (0.55, (255, 150, 70), 0.95),
                       (0.22, (255, 240, 200), 1.0)):
        for i in range(len(pts) - 1):
            w0, w1 = ws[i] * wf / 2, ws[i + 1] * wf / 2
            quad = [pts[i] + nm[i] * w0, pts[i + 1] + nm[i + 1] * w1, pts[i + 1] - nm[i + 1] * w1, pts[i] - nm[i] * w0]
            p = skia.Path()
            p.addPoly([skia.Point(float(x), float(y)) for x, y in quad], True)
            alpha = int(255 * a * fade[i])
            c.drawPath(p, skia.Paint(AntiAlias=True, Color=skia.Color(*col, alpha)))
            ce.drawPath(p, skia.Paint(AntiAlias=True, Color=skia.Color(*col, int(alpha * 0.7))))
    # the bike from behind: a red capsule tail, rider's back, a bar of tail light
    (bx, by), d = cam.project(np.array([xs[0], 0.0, zs[0]]))
    k = cam.f / d
    INK = skia.Color(20, 6, 12)
    def poly(pp, col, cv=c):
        p = skia.Path()
        p.addPoly([skia.Point(float(bx + u * k), float(by - v * k)) for u, v in pp], True)
        cv.drawPath(p, skia.Paint(AntiAlias=True, Color=col))
        c.drawPath(p, skia.Paint(AntiAlias=True, Color=INK, Style=skia.Paint.kStroke_Style, StrokeWidth=0.8))
    poly([(-0.12, 0.0), (0.12, 0.0), (0.14, 0.3), (-0.14, 0.3)], skia.Color(18, 16, 22))                 # tyre
    poly([(-0.36, 0.3), (0.36, 0.3), (0.42, 0.62), (0.3, 0.9), (-0.3, 0.9), (-0.42, 0.62)], skia.Color(150, 24, 36))
    poly([(-0.2, 0.9), (0.2, 0.9), (0.26, 1.35), (0.12, 1.48), (-0.12, 1.48), (-0.26, 1.35)], skia.Color(120, 20, 30))
    poly([(-0.11, 1.48), (0.11, 1.48), (0.1, 1.68), (-0.1, 1.68)], skia.Color(16, 12, 18))
    poly([(-0.3, 0.62), (0.3, 0.62), (0.3, 0.7), (-0.3, 0.7)], skia.Color(255, 90, 60), ce)
    poly([(-0.3, 0.62), (0.3, 0.62), (0.3, 0.7), (-0.3, 0.7)], skia.Color(255, 120, 80))
    a = s.makeImageSnapshot().toarray().astype(np.float32) / 255.0
    b = e.makeImageSnapshot().toarray().astype(np.float32) / 255.0
    a[..., :3] *= a[..., 3:4]
    b[..., :3] *= b[..., 3:4]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
    # photographed light, not a vector stroke: soften the ribbon, keep the core hot
    soft = cv2.GaussianBlur(a, (0, 0), 3.0)
    core = np.clip((a[..., :3].mean(2, keepdims=True) - 0.75) * 4, 0, 1)
    a = soft * 0.85 + a * 0.15
    a[..., :3] += core * 0.35
    b = cv2.GaussianBlur(b, (0, 0), 2.0)
    return a, b[..., :3]


def main(seed=3):
    OUT.mkdir(parents=True, exist_ok=True)
    rs = np.random.default_rng(seed)
    bg, bg_emit = backdrop(rs)
    cam, st = build(rs)
    img, emit = st.render(bg.astype(np.float32), ss=2)
    emit += bg_emit
    light_cone(img, emit, cam)
    road_reflections(img, emit, cam, st)
    dirs = st.stroke_dirs()
    for i, fc in enumerate(st.faces):          # wet ground: the reflections run vertical
        c = fc['c']
        if np.allclose(c[:, 1], c[0, 1]) and c[0, 1] < 0.5:
            dirs[i] = np.pi / 2
    if PAINT_OVER:   # tried in round 5: smeared the crisp edges, judged worse; off
        img = paint_over(np.clip(img, 0, 1.2), st.ids, dirs, seed=seed)
    img = wobble(img, rs, amp=0.35)
    cel = cels(cam, rs)
    img = cel[..., :3] + img * (1 - cel[..., 3:4])
    tr, tre = trail_and_bike(cam, rs)
    img = tr[..., :3] + img * (1 - tr[..., 3:4])
    emit += tre
    out = develop(img, emit, seed=seed, lens=0.75, texbank=True)
    Image.fromarray((out * 255 + .5).astype(np.uint8)).save(OUT / 'o03_street.png')
    print('wrote', OUT / 'o03_street.png')


if __name__ == '__main__':
    main()
