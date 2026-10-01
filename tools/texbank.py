"""Film texture bank: the paint tooth and grain of the real print, without content.

From frames of the film, take only patches that are truly flat (no structure
in a smoothed copy), and keep the residual against that smooth copy: grain,
pigment tooth, print speckle. Patches with outliers (a speck of detail, a
line) are rejected, so no drawing leaks through. Saved to out/texbank.npz
(derived from film frames, so it stays local and gitignored).

    python tools/texbank.py            # samples ref/akira_film/*.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = 48


def main(max_frames=600, max_patches=30000, seed=0):
    rs = np.random.default_rng(seed)
    files = sorted((ROOT / 'ref' / 'akira_film').glob('f_*.jpg'))
    files = [files[i] for i in rs.choice(len(files), min(max_frames, len(files)), replace=False)]
    patches, levels = [], []
    for f in files:
        im = cv2.imread(str(f))[..., ::-1].astype(np.float32)
        if im.mean() < 12 or im.std() < 8:
            continue  # black frames, fades
        smooth = cv2.GaussianBlur(im, (0, 0), 2.5)
        res = im - smooth
        g = smooth.mean(2)
        grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)) / 8
        h, w = g.shape
        for y in range(8, h - P - 8, P):
            for x in range(8, w - P - 8, P):
                if grad[y:y + P, x:x + P].max() > 1.2:
                    continue
                r = res[y:y + P, x:x + P]
                sd = r.std()
                if sd < 0.4 or np.abs(r).max() > 6 * sd + 2:
                    continue
                patches.append(r)
                levels.append(g[y:y + P, x:x + P].mean() / 255.0)
        if len(patches) > max_patches:
            break
    patches = np.array(patches, np.float32)
    levels = np.array(levels, np.float32)
    np.savez_compressed(ROOT / 'out' / 'texbank.npz', patches=patches, levels=levels)
    print(len(patches), 'patches from', len(files), 'frames; residual std', patches.std(axis=(0, 1, 2)).round(2))
    for lo in (0, .1, .25, .5, .75):
        sel = (levels >= lo) & (levels < lo + .15)
        if sel.any():
            print(f'  level {lo:.2f}: n={sel.sum()} std={patches[sel].std():.2f}')


if __name__ == '__main__':
    main()
