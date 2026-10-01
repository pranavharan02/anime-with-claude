"""Recreate real Akira frames one-to-one as code.

    python recreate.py 048 062 --k 48

Reads ref/akira/akira_NNN.jpg (gitignored), writes to out/akira_frames/:
  akira_NNN.json       the frame as code: airbrush mesh, cel paths, ink strokes
  akira_NNN_code.png   that code rendered by Skia, plus matched film grain
  akira_NNN_2x.png     the same code rendered at 2x (it is vector)
  akira_NNN_cmp.png    reference | code render | colour-error heat map
and prints SSIM / PSNR / mean delta-E against the reference.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from engine.trace import vectorize
from engine.scene import to_json, render, refine

REF = ROOT / 'ref' / 'akira'
OUT = ROOT / 'out' / 'akira_frames'


def grain_sigma(img):
    """Film grain strength in the reference: residual after a light denoise."""
    f = img.astype(np.float32)
    smooth = cv2.bilateralFilter(f, 7, 20, 5)
    return float(np.median(np.abs(f - smooth)) * 1.4826)


def add_grain(rgb, sigma, seed=0):
    rs = np.random.default_rng(seed)
    n = rs.standard_normal(rgb.shape[:2] + (1,)).astype(np.float32)
    n = n + 0.35 * rs.standard_normal(rgb.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6)
    n /= n.std() + 1e-6
    return np.clip(rgb + n * sigma / 255.0, 0, 1)


def score(ref, out, fast=False):
    a, b = ref.astype(np.float32) / 255.0, out
    ssim = structural_similarity(a, b, channel_axis=2, data_range=1.0)
    if fast:
        return ssim
    psnr = peak_signal_noise_ratio(a, b, data_range=1.0)
    de = deltaE_ciede2000(rgb2lab(a), rgb2lab(b))
    return ssim, psnr, float(de.mean()), de


def heat(de):
    v = np.clip(de / 20.0, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def run(fid, k, min_area, eps, refine_iters=4):
    src = REF / f'akira_{fid}.jpg'
    ref = np.array(Image.open(src).convert('RGB'))
    h, w = ref.shape[:2]
    t0 = time.time()
    sc = vectorize(ref, k=k, min_area=min_area, eps=eps)
    js = to_json(sc)
    (OUT / f'akira_{fid}.json').write_text(js, encoding='utf8')
    scene = json.loads(js)
    code = render(scene)
    # camera: fit the lens softness of the original print to this frame
    lens = max((0.0, 0.4, 0.6, 0.8, 1.0, 1.2),
               key=lambda sg: score(ref, cv2.GaussianBlur(code, (0, 0), sg) if sg else code, fast=True))
    if refine_iters:
        pix = [f[4] for f in sc['fills']] + [l[3] for l in sc['lines']]
        scene = refine(scene, pix, ref, iters=refine_iters, lens=lens)
        js = json.dumps(scene, separators=(',', ':'))
    (OUT / f'akira_{fid}.json').write_text(js, encoding='utf8')
    scene = json.loads(js)          # render from the saved code, not from memory
    code = render(scene)
    if lens:
        code = cv2.GaussianBlur(code, (0, 0), lens)
    layers = [render(scene, layers=l) for l in (('mesh',), ('mesh', 'fills'), ('lines',))]
    strip = np.concatenate([(x * 255 + .5).astype(np.uint8) for x in layers], 1)
    Image.fromarray(strip).save(OUT / f'akira_{fid}_layers.png')
    ssim, psnr, de_mean, de = score(ref, code)
    g = grain_sigma(ref)
    filmed = add_grain(code, g, seed=int(fid))
    Image.fromarray((filmed * 255 + .5).astype(np.uint8)).save(OUT / f'akira_{fid}_code.png')
    big = render(scene, scale=2.0)
    Image.fromarray((big * 255 + .5).astype(np.uint8)).save(OUT / f'akira_{fid}_2x.png')
    cmp_ = np.concatenate([ref, (filmed * 255 + .5).astype(np.uint8), heat(de)], 1)
    Image.fromarray(cmp_).save(OUT / f'akira_{fid}_cmp.png')
    grads = sum(1 for f in sc['fills'] if f[1][0] == 'lin')
    feathered = sum(1 for f in sc['fills'] if f[3] > 0)
    res = dict(frame=fid, fills=len(sc['fills']), gradients=grads, feathered=feathered, lines=len(sc['lines']),
               airbrush=round(sc['soft_frac'], 2), code_kb=round(len(js) / 1024),
               ssim=round(float(ssim), 4), psnr=round(float(psnr), 2), mean_dE=round(de_mean, 2),
               grain=round(g, 2), lens=lens, secs=round(time.time() - t0, 1), k=k, min_area=min_area, eps=eps, refine=refine_iters)
    print(json.dumps(res))
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('frames', nargs='+')
    ap.add_argument('--k', type=int, default=128)
    ap.add_argument('--min-area', type=int, default=4)
    ap.add_argument('--eps', type=float, default=0.3)
    ap.add_argument('--refine', type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in a.frames:
        run(f, a.k, a.min_area, a.eps, a.refine)
