"""Plate 02 + motion test m01: the bike at speed with its tail-light trail.

Camera tracks the bike. The city slides past in three parallax strips with
speed blur. The trail is the tail light's own position history, so it waves
the way the bike actually moved.

    python p02_trail.py            # still plate
    python p02_trail.py --motion   # 3 s at 24 fps, animated on twos
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import skia

import bike
from engine.core import W, H, surface, to_np, flatten, save, col, hx, mix, over
from engine.city import City, NIGHT, rect
from engine.fx import bloom, lin_grad, ribbon, blur
from engine.ink import path_of, curve
from engine.paint import gouache
from engine.film import develop

OUT = ROOT / 'out' / 'akira'
FPS, DUR = 24, 3.0
SPEED = 1500.0                       # px/s at the bike's depth
LAYERS = ((0.18, 0.95), (0.45, 0.7), (0.85, 0.42))   # (parallax factor, depth)
ROAD_Y = 880
SCALE = 0.95
BIKE_X, BIKE_Y = 760.0, 990 - 585 * SCALE


def motion_blur(a, length):
    n = max(1, int(length))
    if n < 2:
        return a
    k = np.ones((1, n), np.float32) / n
    return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT)


def build_strips(seed=5):
    """Paint the three city strips once; return list of (factor, paint, emit)."""
    strips = []
    for i, (factor, depth) in enumerate(LAYERS):
        sw = int(W + factor * SPEED * DUR) + 200
        ps, es = surface(sw, H), surface(sw, H)
        city = City(ps.getCanvas(), es.getCanvas(), seed=seed + i)
        base = ROAD_Y - 40 + 30 * i
        hmin, hmax = [(250, 560), (200, 520), (260, 760)][i]
        city.row(-100, sw + 100, base, hmin, hmax, depth, density=1.1 + 0.2 * (1 - i))
        city.haze(base - 220, base, a1=0.55 - 0.15 * i, w=sw)
        paint = to_np(ps)
        paint[..., :3] = gouache(paint[..., :3], seed + i, amt=0.04)
        emit = to_np(es)[..., :3]
        blur_len = factor * SPEED / FPS * 1.6   # shutter open across ~1.6 frames
        strips.append((factor, motion_blur(paint, blur_len), motion_blur(emit, blur_len)))
    return strips


def sky():
    s = surface()
    lin_grad(s.getCanvas(), path_of(rect(0, 0, W, H)), (0, 0), (0, ROAD_Y),
             [NIGHT['sky_top'], NIGHT['sky_mid'], NIGHT['sky_low']], [0, 0.6, 1])
    return flatten(to_np(s))


def road(offset):
    """Wet expressway deck with lane dashes and a lit guard rail, blurred."""
    ps, es = surface(), surface()
    c, e = ps.getCanvas(), es.getCanvas()
    lin_grad(c, path_of(rect(0, ROAD_Y - 6, W, H - ROAD_Y + 6)), (0, ROAD_Y), (0, H),
             ['#2a4a55', '#14232e', '#0c161f'], [0, 0.25, 1])
    # guard rail with lamps
    c.drawRect(skia.Rect(0, ROAD_Y - 26, W, ROAD_Y - 6), skia.Paint(Color4f=col('#10202a')))
    c.drawRect(skia.Rect(0, ROAD_Y - 26, W, ROAD_Y - 22), skia.Paint(Color4f=col('#5fb8b4')))
    step = 260
    for k in range(-1, W // step + 2):
        x = k * step - (offset * 1.0) % step
        e.drawRect(skia.Rect(x, ROAD_Y - 30, x + 18, ROAD_Y - 24), skia.Paint(Color4f=col('#ffd890')))
        c.drawRect(skia.Rect(x, ROAD_Y - 30, x + 18, ROAD_Y - 24), skia.Paint(Color4f=col('#ffe7b0')))
    # lane dashes (nearer than the bike: they move faster)
    step = 300
    for k in range(-1, W // step + 2):
        x = k * step - (offset * 1.25) % step
        c.drawRect(skia.Rect(x, 1012, x + 140, 1018), skia.Paint(Color4f=col('#8fb3b8', 0.7)))
    paint, emit = flatten(to_np(ps)), to_np(es)[..., :3]
    paint[ROAD_Y - 32:] = motion_blur(paint[ROAD_Y - 32:], SPEED / FPS * 1.4)
    emit = motion_blur(emit, SPEED / FPS * 1.4)
    return paint, emit


def tail_y(t):
    """Vertical weave of the tail light (screen px) over time."""
    return 16 * np.sin(2 * np.pi * 1.05 * t) + 6 * np.sin(2 * np.pi * 2.3 * t + 1.0)


def trail(tx, ty0, t, length=2300):
    """Ribbon through the tail light's past positions -> (paint RGBA, emit RGB)."""
    d = np.linspace(0, length, 400)
    ys = ty0 + tail_y(t - d / SPEED) - tail_y(t)
    pts = np.stack([tx - d, ys], 1)
    u = d / length
    w_body = 40 * (0.45 + 0.55 * np.minimum(1, u * 8)) * (1 - 0.3 * u)
    ps, es = surface(), surface()
    c, e = ps.getCanvas(), es.getCanvas()
    for wf, colr, a in ((2.4, '#ff1e14', 0.3), (1.0, '#ff3418', 1.0),
                        (0.64, '#ff8a30', 1.0), (0.3, '#fff2c0', 1.0)):
        poly = path_of(ribbon(pts, w_body * wf))
        c.drawPath(poly, skia.Paint(AntiAlias=True, Color4f=col(colr, a)))
        e.drawPath(poly, skia.Paint(AntiAlias=True, Color4f=col(colr, a * 0.8)))
    # fade with distance from the tail light (per pixel, so no seams)
    xs = np.arange(W, dtype=np.float32)
    fade = np.clip(1 - (tx - xs) / length, 0, 1) ** 1.2
    fade[xs > tx] = 1
    paint, emit = to_np(ps), to_np(es)[..., :3]
    return paint * fade[None, :, None], emit * fade[None, :, None]


def frame(t, strips, sky_rgb, fidx=0, seed=0):
    img = sky_rgb.copy()
    emit = np.zeros((H, W, 3), np.float32)
    for factor, paint, em in strips:
        ox = int(factor * SPEED * t)
        img = flatten(paint[:, ox:ox + W], img)
        emit += em[:, ox:ox + W]
    rp, re = road(SPEED * t)
    img[ROAD_Y - 32:] = rp[ROAD_Y - 32:]
    emit += re

    # cel layer: trail (in the air behind the bike), then the bike itself
    cs, es = surface(), surface()
    c, e = cs.getCanvas(), es.getCanvas()
    f2 = int(t * FPS) // 2          # animate the cel on twos
    t2 = f2 * 2 / FPS
    bob = tail_y(t2) / SCALE
    ty = BIKE_Y + (380 + bob) * SCALE
    tx = BIKE_X + 46 * SCALE
    tp, te = trail(tx, ty, t2)
    # reflection of the trail on the wet road: flipped, smeared, dim
    refl = motion_blur(blur(tp[::-1], 5), 60)
    shift = int(2 * 1000 - H)
    refl = np.roll(refl, shift - 30, axis=0) * 0.3
    refl[:ROAD_Y] = 0
    img = flatten(refl, img)
    img = flatten(tp, img)
    emit += te * 1.3
    bike.draw(c, e, BIKE_X, BIKE_Y + bob * SCALE, SCALE, spin=-t2 * 40, seed=f2)

    cel = to_np(cs)
    # night cel grade: cels sit a touch darker and cooler than the BG
    img = flatten(cel, img)
    emit += to_np(es)[..., :3]
    img = img + bloom(emit, ((3, .5), (10, .55), (28, .55), (80, .45)))
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    strips = build_strips()
    sk = sky()
    if '--motion' in sys.argv:
        fdir = OUT / 'm01_frames'
        fdir.mkdir(exist_ok=True)
        n = int(FPS * DUR)
        for i in range(n):
            img = frame(i / FPS, strips, sk, i)
            save(develop(img, frame=i, seed=2), fdir / f'{i:04d}.png')
            print(f'frame {i + 1}/{n}', end='\r')
        mp4 = OUT / 'm01_trail.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i',
                        str(fdir / '%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '16', '-vf', 'pad=1920:1040:0:1', str(mp4)], check=True)
        print('\nwrote', mp4)
    else:
        img = frame(1.1, strips, sk)
        save(develop(img, frame=3, seed=2), OUT / 'p02_trail.png')
        print('wrote', OUT / 'p02_trail.png')


if __name__ == '__main__':
    main()
