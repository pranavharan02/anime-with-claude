"""Blind-test harness: candidate frame(s) shuffled among real Akira frames.

    python tools/blind.py ROUND out/akira_originals/o03_street.png [more.png ...]

Writes out/blind/<ROUND>/still_N.jpg (all 1280x718, same JPEG quality) and a
hidden key at out/blind/<ROUND>_key.json. A fresh judge agent is then pointed
at the folder only. Real frames are drawn at random from ref/akira.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    args = sys.argv[1:]
    n_real, cats = 3, None
    while args and args[0].startswith('--'):
        k = args.pop(0)
        if k == '--reals':
            n_real = int(args.pop(0))
        elif k == '--cats':          # draw reals from these city_frames.json categories
            cats = args.pop(0).split(',')
    rnd, cands = args[0], args[1:]
    rs = random.Random(rnd)
    film = sorted((ROOT / 'ref' / 'akira_film').glob('f_*.jpg'))
    pool = film if len(film) > 100 else sorted((ROOT / 'ref' / 'akira').glob('akira_*.jpg'))
    if cats:
        cf = json.loads((ROOT / 'out' / 'study' / 'city_frames.json').read_text())
        ids = {int(i) for c in cats for i in cf.get(c, [])}
        pool = [p for p in film if int(p.stem.split('_')[1]) in ids]
    reals = []
    while len(reals) < n_real:   # skip fades, black frames and title cards
        p = rs.choice(pool)
        a = np.asarray(Image.open(p).convert('L'), np.float32)
        if a.mean() > 25 and a.std() > 22 and p not in reals:
            reals.append(p)
    items = [('real', str(p)) for p in reals] + [('cand', c) for c in cands]
    rs.shuffle(items)
    d = ROOT / 'out' / 'blind' / rnd
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob('*'):
        f.unlink()
    key = {}
    for i, (kind, p) in enumerate(items, 1):
        im = Image.open(p).convert('RGB')
        if kind == 'real' and 'akira_film' in str(p):      # crop the film's letterbox bars
            im = im.crop((0, 14, im.width, im.height - 14))
        im = im.resize((1280, 718), Image.LANCZOS)
        name = f'still_{i}.jpg'
        im.save(d / name, quality=88)
        key[name] = f'{kind}:{Path(p).name}'
    (ROOT / 'out' / 'blind' / f'{rnd}_key.json').write_text(json.dumps(key, indent=1))
    print(d, len(items))


if __name__ == '__main__':
    main()
