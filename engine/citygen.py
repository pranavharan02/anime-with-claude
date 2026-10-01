"""An original megacity, generated once, shot from many cameras.

Follows refs/akira/CITY_BIBLE.md (rule numbers in brackets). The rules are
about form and painting; no building is copied from any film.

  strata [1]   old city (3-8 storeys, grimy, shop fronts), 1980s office slabs
               (15-40 storeys) in a ring, megatowers (80-150+ storeys,
               slender) massed in one core [3]
  megatowers   2-4 stacked prisms (box, chamfered octagon, drum), setbacks
               near 1/3 and 2/3, belt courses, a crown, a spire; each tower
               its own hue family, warm and cool alternating [2, 18]
  streets      narrow slots; elevated expressways on teal-green piers [5, 6]
  roofs        plant on every roof, a red beacon on every top [7]

populate() instantiates only what a camera sees, paints each face at its
distance class, and applies discrete haze planes that glow from the base [21, 31].
"""
import numpy as np

from . import gouache as G
from .facade import paint_facade

FAMILIES = {  # wall (lit face), shade (shadow face: magenta-purple leaning), windows, rim [18, 13]
    'teal': ('#123f40', '#141a28', ['#e8dca0', '#b8c890', '#d8e8c0'], '#4fc0b0'),
    'plum': ('#3a2434', '#1a1222', ['#e8dca0', '#f2d8b0', '#8a98a8'], '#d890b8'),
    'indigo': ('#2a4258', '#161e32', ['#e8dca0', '#cfe0f0', '#8a98a8'], '#8ac8e8'),
    'jade': ('#173a32', '#121622', ['#e8dca0', '#b8c890', '#f0e4b0'], '#6ad0a0'),
    'slate': ('#222c3a', '#141220', ['#e8dca0', '#b8c890', '#8a98a8'], None),
    'rust': ('#3a2620', '#1c141c', ['#f0e0a8', '#e8c890', '#b8c890'], '#e0a070'),
}
OLD_WALLS = [('#4a4650', '#2a2232'), ('#3e4a48', '#22202e'), ('#52463e', '#30222a'), ('#4a5048', '#262230'),
             ('#5a4a4a', '#34262e')]
SLAB_WALLS = [('#2e3a48', '#1e1a2a'), ('#2a3e3c', '#1a1c28'), ('#3a3644', '#221a2a'), ('#30403a', '#1c1e2a')]


class City:
    def __init__(self, seed=1, half_x=1500, z0=0, z1=3400, core=(0.0, 1700.0)):
        self.rs = np.random.default_rng(seed)
        self.core = np.array(core, float)
        self.parts, self.ground, self.decks, self.lamps, self.beacons = [], [], [], [], []
        rs = self.rs
        ax = [-half_x]
        while ax[-1] < half_x:
            ax.append(ax[-1] + rs.uniform(90, 130))
        sz = [z0]
        while sz[-1] < z1:
            sz.append(sz[-1] + rs.uniform(60, 95))
        self.avenues, self.streets = ax, sz
        self.aw = {x: (24 if i % 3 == 0 else 14) for i, x in enumerate(ax)}
        self.sw = {z: (16 if i % 4 == 0 else 10) for i, z in enumerate(sz)}
        for x in ax:
            w = self.aw[x]
            self.ground.append((x - w / 2, x + w / 2, z0, z1, 'road'))
            for k in range(int((z1 - z0) / 35)):
                self.lamps.append((x - w / 2 + 1.2, 8.0, z0 + k * 35))
                self.lamps.append((x + w / 2 - 1.2, 8.0, z0 + k * 35 + 17))
        for z in sz:
            self.ground.append((-half_x, half_x, z - self.sw[z] / 2, z + self.sw[z] / 2, 'road'))
        mid = len(ax) // 2
        for i in (mid - 3, mid + 2):
            self.decks.append(dict(x=ax[i], w=20.0, z0=z0, z1=z1, y=float(rs.uniform(15, 20))))
            for k in range(int((z1 - z0) / 30)):
                self.lamps.append((ax[i] + 9.4, self.decks[-1]['y'] + 7, z0 + k * 30))
        self._fam_i = int(rs.integers(len(FAMILIES)))
        for i in range(len(ax) - 1):
            for j in range(len(sz) - 1):
                bx0, bx1 = ax[i] + self.aw[ax[i]] / 2 + 3, ax[i + 1] - self.aw[ax[i + 1]] / 2 - 3
                bz0, bz1 = sz[j] + self.sw[sz[j]] / 2 + 3, sz[j + 1] - self.sw[sz[j + 1]] / 2 - 3
                if bx1 - bx0 < 18 or bz1 - bz0 < 18:
                    continue
                self.ground.append((bx0 - 3, bx1 + 3, bz0 - 3, bz1 + 3, 'walk'))
                self._block(bx0, bx1, bz0, bz1)

    # -- zoning -------------------------------------------------------------------
    def _zone(self, x, z):
        d = np.hypot(x - self.core[0], (z - self.core[1]) * 0.85)
        if d < 380:
            return 'core', d
        if d < 1000:
            return 'ring', d
        return 'old', d

    def _block(self, x0, x1, z0, z1):
        rs = self.rs
        zone, d = self._zone((x0 + x1) / 2, (z0 + z1) / 2)
        if zone == 'core':
            self._megatower(x0, x1, z0, z1, d)
            self._old(x0, x1, z0, z0 + 12)
            self._old(x0, x1, z1 - 12, z1)
        elif zone == 'ring' and rs.random() < 0.75 - 0.4 * (d - 380) / 620:
            self._slab_block(x0, x1, z0, z1, d)
        else:
            for lot in self._split(x0, x1, z0, z1, rs.integers(3, 7)):
                self._old(*lot)

    # -- parts --------------------------------------------------------------------
    def _box(self, x0, x1, z0, z1, y0, y1, mat, roof='#0d1218'):
        self.parts.append(dict(kind='box', b=(x0, x1, y0, y1, z0, z1), mat=mat, roof=roof))

    def _prism(self, cx, cz, r, n, y0, y1, mat, rot=0.0, roof='#0d1218'):
        self.parts.append(dict(kind='prism', c=(cx, cz), r=r, n=n, y=(y0, y1), mat=mat, rot=rot, roof=roof))

    def _plant(self, x0, x1, z0, z1, y, n=None):
        """Roof clutter: tanks, plant rooms, a mast [7]."""
        rs = self.rs
        if x1 - x0 < 8 or z1 - z0 < 8:
            return
        m = dict(wall='#2a2e36', shade='#181b20', style='none')
        for _ in range(int(n or rs.integers(5, 10))):
            w, d = rs.uniform(2.5, max(3, 0.3 * (x1 - x0))), rs.uniform(2.5, max(3, 0.3 * (z1 - z0)))
            px, pz = rs.uniform(x0, max(x0 + 0.1, x1 - w)), rs.uniform(z0, max(z0 + 0.1, z1 - d))
            if rs.random() < 0.35:
                self._prism(px + w / 2, pz + w / 2, w / 2, 8, y, y + rs.uniform(2, 5), dict(m, wall='#3a3e46'))
            else:
                self._box(px, px + w, pz, pz + d, y, y + rs.uniform(2, 6), m)
        if rs.random() < 0.35:
            mx, mz = rs.uniform(x0, x1), rs.uniform(z0, z1)
            self._box(mx - 0.4, mx + 0.4, mz - 0.4, mz + 0.4, y, y + rs.uniform(10, 25), m)

    def _megatower(self, x0, x1, z0, z1, d):
        rs = self.rs
        names = list(FAMILIES)
        wall, shade, pal, rim = FAMILIES[names[self._fam_i % len(names)]]
        self._fam_i += 1 + int(rs.integers(0, 2))
        h = (620 - d * 0.9) * rs.uniform(0.75, 1.1)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        r = min(x1 - x0, z1 - z0) * rs.uniform(0.36, 0.46)
        mat = dict(wall=wall, shade=shade, pal=pal, rim=rim, rim_side=str(rs.choice(['left', 'right'])),
                   floor=float(rs.choice([3.6, 4.0, 4.2])) * rs.uniform(0.92, 1.08), bay=rs.uniform(2.2, 3.4),
                   lit=rs.uniform(0.45, 0.7), belt=int(rs.integers(8, 16)), belt_col=pal[rs.integers(len(pal))],
                   style='grid')
        n_st = int(rs.integers(2, 5))
        cuts = np.sort(np.clip(np.array([1 / 3, 2 / 3, 0.85])[:n_st - 1] + rs.normal(0, 0.05, n_st - 1), 0.2, 0.92))
        bounds = np.concatenate([[0], cuts, [1]]) * h
        for k in range(n_st):
            y0, y1 = bounds[k], bounds[k + 1]
            shape = rs.choice(['box', 'oct', 'drum'])
            if shape == 'drum':
                self._prism(cx, cz, r, 16, y0, y1, dict(mat))
            elif shape == 'oct':
                self._prism(cx, cz, r, 8, y0, y1, dict(mat), rot=np.pi / 8)
            else:
                self._box(cx - r, cx + r, cz - r, cz + r, y0, y1, dict(mat))
            r *= rs.uniform(0.7, 0.86)
        top = bounds[-1]
        crown = rs.choice(['steps', 'lid', 'twin', 'antennas'], p=[0.3, 0.3, 0.2, 0.2])
        if crown == 'twin':                          # paired slabs joined by a bridge [2]
            g = r * 0.25
            th_ = h * rs.uniform(0.12, 0.2)
            self._box(cx - r, cx - g, cz - r * 0.6, cz + r * 0.6, top, top + th_, dict(mat))
            self._box(cx + g, cx + r, cz - r * 0.6, cz + r * 0.6, top, top + th_ * rs.uniform(0.8, 1.2), dict(mat))
            self._box(cx - g, cx + g, cz - r * 0.3, cz + r * 0.3, top + th_ * 0.55, top + th_ * 0.7,
                      dict(mat, style='none', wall=mat['belt_col']))
            top += th_
        elif crown == 'antennas':                    # a cluster of lattice masts
            for _ in range(int(rs.integers(3, 6))):
                ax_, az_ = cx + rs.uniform(-r, r) * 0.7, cz + rs.uniform(-r, r) * 0.7
                hh_ = h * rs.uniform(0.04, 0.12)
                self._box(ax_ - 0.5, ax_ + 0.5, az_ - 0.5, az_ + 0.5, top, top + hh_,
                          dict(wall='#3a3e46', shade='#22252b', style='none'))
                self.beacons.append((ax_, top + hh_, az_))
        elif crown == 'steps':                       # stepped cap
            for s in range(3):
                rr = r * (1 - 0.25 * s)
                self._box(cx - rr, cx + rr, cz - rr, cz + rr, top, top + 6, dict(mat, style='none', belt=0))
                top += 6
        else:                                        # flared lid
            self._prism(cx, cz, r * 1.25, 12, top, top + 8, dict(mat, style='none', belt=0, wall=mat['belt_col']))
            top += 8
        sp = h * rs.uniform(0.05, 0.15)
        self._box(cx - 0.8, cx + 0.8, cz - 0.8, cz + 0.8, top, top + sp,
                  dict(wall='#3a3e46', shade='#22252b', style='none'))
        self.beacons.append((cx, top + sp, cz))
        self._plant(cx - r, cx + r, cz - r, cz + r, top)

    def _slab_block(self, x0, x1, z0, z1, d):
        rs = self.rs
        for a, b, c, e in self._split(x0, x1, z0, z1, rs.integers(1, 3)):
            wall, shade = SLAB_WALLS[rs.integers(len(SLAB_WALLS))]
            h = rs.uniform(55, 150) * (1.2 - 0.5 * (d - 380) / 620)
            mat = dict(wall=wall, shade=shade, pal=['#e8dca0', '#b8c890', '#8a98a8', '#f0e4b0'], floor=3.6,
                       bay=rs.uniform(2.4, 3.2), lit=rs.uniform(0.55, 0.8), style='grid', rim=None, grime=0.15)
            self._box(a, b, c, e, 0, 8, dict(mat, floor=4.5, lit=0.7, pal=['#f0e2b0']))   # colonnade base
            self._box(a + 1, b - 1, c + 1, e - 1, 8, h, mat)
            self._plant(a + 2, b - 2, c + 2, e - 2, h)
            pass

    def _old(self, x0, x1, z0, z1):
        rs = self.rs
        x = x0
        while x < x1 - 3:
            w = min(x1 - x, rs.uniform(6, 15))
            h = int(rs.integers(3, 9)) * 3.1
            wall, shade = OLD_WALLS[rs.integers(len(OLD_WALLS))]
            mat = dict(wall=wall, shade=shade, pal=['#f0e0a8', '#e8c890', '#b8c890', '#8a98a8'], floor=3.1,
                       bay=rs.uniform(1.8, 2.6), lit=rs.uniform(0.3, 0.6), style='grid', grime=0.9, shop=True, rim=None)
            self._box(x, x + w - 0.4, z0, z1, 0, h, mat, roof='#181a1e')
            self._plant(x + 1, x + w - 2, z0 + 1, z1 - 1, h, n=rs.integers(1, 4))
            x += w

    def _split(self, x0, x1, z0, z1, n):
        rs = self.rs
        lots = [(x0, x1, z0, z1)]
        for _ in range(int(n)):
            lots.sort(key=lambda l: -(l[1] - l[0]) * (l[3] - l[2]))
            a, b, c, d = lots.pop(0)
            if (b - a) >= (d - c) and b - a > 24:
                m = rs.uniform(a + 0.35 * (b - a), a + 0.65 * (b - a))
                lots += [(a, m - 1, c, d), (m + 1, b, c, d)]
            elif d - c > 24:
                m = rs.uniform(c + 0.35 * (d - c), c + 0.65 * (d - c))
                lots += [(a, b, c, m - 1), (a, b, m + 1, d)]
            else:
                lots.append((a, b, c, d))
        return lots

    # -- per-shot instantiation ----------------------------------------------------
    def populate(self, st, cam, ss=2, quality=1.0, max_dist=6000, seed=0,
                 haze=('#826f82', (250, 600, 1200, 2200)), key=(0.55, 0.0, -0.84)):
        rs = np.random.default_rng(seed)
        W, H, eye = cam.w, cam.h, cam.eye
        hz_col = G.hx(haze[0])
        key = np.array(key) / np.linalg.norm(key)

        def visible(corners):
            c = cam.to_cam(np.asarray(corners, float))
            if (c[:, 2] <= 0.5).any():
                return False
            p, _ = cam.project(np.asarray(corners, float))
            return not (p[:, 0].max() < -2 or p[:, 0].min() > W + 2 or p[:, 1].max() < -2 or p[:, 1].min() > H + 2)

        def hazed(tex, emit, dist, y0):
            a = [0.0, 0.12, 0.26, 0.42, 0.58][min(int(np.searchsorted(haze[1], dist)), 4)]
            if a:
                v = np.linspace(0, 1, tex.shape[0], dtype=np.float32)[:, None, None]
                g = a * (0.55 + 0.6 * v ** 2) if y0 < 30 else a * 0.55   # glow rises from the street
                tex = tex * (1 - g) + hz_col * g
                emit = emit * (1 - 0.6 * a)
            return tex, emit

        def add_face(corners, mat, wm, hm, lit_face, y0):
            corners = np.asarray(corners, float)
            if not visible(corners):
                return
            dist = np.linalg.norm(corners.mean(0) - eye)
            if dist > max_dist:
                return
            if dist > 1500 and hm < 60:     # far low-rise: a dark mass with a few lights
                mat = dict(mat, wall=G.hx(mat['wall']) * 0.45, shade=G.hx(mat['shade']) * 0.45,
                           lit=mat.get('lit', 0.4) * 0.35, shop=False, grime=0)
            ppm = cam.f * ss / max(dist, 1) * quality
            bay_px = mat.get('bay', 2.6) * cam.f / max(dist, 1)
            t, e = paint_facade(mat, wm, hm, ppm, rs, bay_px=bay_px, lit_face=lit_face)
            t, e = hazed(t, e, dist, y0)
            st.face(corners, t, emit=e, lit=False)

        def add_roof(corners, col, y):
            corners = np.asarray(corners, float)
            if eye[1] <= y or not visible(corners):
                return
            dist = np.linalg.norm(corners.mean(0) - eye)
            if dist > max_dist:
                return
            ppm = min(cam.f * ss / max(dist, 1), 6)
            w = np.linalg.norm(corners[1] - corners[0])
            d = np.linalg.norm(corners[3] - corners[0])
            th_, tw_ = max(2, int(d * ppm)), max(2, int(w * ppm))
            t = np.empty((th_, tw_, 3), np.float32)
            t[:] = G.hx(col)
            if th_ > 6 and tw_ > 6:   # roofs: stains, a lighter parapet ring, sponge texture
                t *= (1 + 0.12 * G.field(rs, th_, tw_, max(2, th_ / 6), max(2, tw_ / 6)))[..., None]
                G.stains(t, rs, max(1, th_ * tw_ // 4000), '#05070a', size=(3, max(4, tw_ / 6)), alpha=(0.2, 0.5))
                b_ = max(1, int(ppm * 0.4))
                t[:b_] = t[-b_:] = t[:, :b_] = t[:, -b_:] = G.hx(col) * 2.2
            t, _ = hazed(t, np.zeros_like(t), dist, y)
            st.face(corners, t, lit=False)

        def lean(q, y0, y1, k, centre=None):
            """Hand-drafted geometry: a building sits a degree or so off the grid
            and leans a fraction of a degree, so nothing converges perfectly."""
            q = np.array(q, float)
            lr = np.random.default_rng(k * 7919 + 13)
            if centre is not None:
                a = np.radians(lr.normal(0, 1.1))
                cx_, cz_ = centre
                dx, dz = q[:, 0] - cx_, q[:, 2] - cz_
                q[:, 0] = cx_ + dx * np.cos(a) - dz * np.sin(a)
                q[:, 2] = cz_ + dx * np.sin(a) + dz * np.cos(a)
            lx, lz = lr.normal(0, 0.007), lr.normal(0, 0.007)
            return q + (q[:, 1] - y0)[:, None] * np.array([[lx, 0, lz]])

        for k_part, pt in enumerate(self.parts):
            if pt['kind'] == 'box':
                x0, x1, y0, y1, z0, z1 = pt['b']
                quads = []
                if eye[2] < z0:
                    quads.append(([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], x1 - x0, (0, 0, -1)))
                if eye[2] > z1:
                    quads.append(([(x1, y1, z1), (x0, y1, z1), (x0, y0, z1), (x1, y0, z1)], x1 - x0, (0, 0, 1)))
                if eye[0] < x0:
                    quads.append(([(x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1)], z1 - z0, (-1, 0, 0)))
                if eye[0] > x1:
                    quads.append(([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], z1 - z0, (1, 0, 0)))
                for q, wm, n in quads:
                    add_face(lean(q, y0, y1, k_part, ((x0 + x1) / 2, (z0 + z1) / 2)), pt['mat'], wm, y1 - y0,
                             float(np.dot(n, key)) > 0.0, y0)
                add_roof(lean([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], y0, y1, k_part,
                              ((x0 + x1) / 2, (z0 + z1) / 2)), pt['roof'], y1)
            else:
                cx, cz = pt['c']
                r, n, rot = pt['r'], pt['n'], pt.get('rot', 0.0)
                y0, y1 = pt['y']
                ring = []
                for i in range(n):
                    a0, a1 = rot + 2 * np.pi * i / n, rot + 2 * np.pi * (i + 1) / n
                    p0 = (cx + r * np.cos(a0), cz + r * np.sin(a0))
                    p1 = (cx + r * np.cos(a1), cz + r * np.sin(a1))
                    ring.append(p0)
                    q = np.array([(p1[0], y1, p1[1]), (p0[0], y1, p0[1]), (p0[0], y0, p0[1]), (p1[0], y0, p1[1])])
                    nrm = np.array([np.cos((a0 + a1) / 2), 0, np.sin((a0 + a1) / 2)])
                    if np.dot(eye - q.mean(0), nrm) <= 0:
                        continue
                    add_face(lean(q, y0, y1, k_part), pt['mat'], np.linalg.norm(q[1] - q[0]), y1 - y0,
                             float(np.dot(nrm, key)) > -0.2, y0)
                if eye[1] > y1:
                    for i in range(0, n, 2):
                        a, b_, c_ = ring[i], ring[(i + 1) % n], ring[(i + 2) % n]
                        add_roof([(cx, y1, cz), (a[0], y1, a[1]), (b_[0], y1, b_[1]), (c_[0], y1, c_[1])], pt['roof'], y1)
        self._ground(st, cam, ss, rs, visible, hazed)
        self._expressways(st, cam, ss, visible, hazed)

    def _ground(self, st, cam, ss, rs, visible, hazed):
        eye = cam.eye
        tiles = []
        for x0, x1, z0, z1, kind in self.ground:
            zz = z0
            while zz < z1:
                zb = min(z1, zz + 50.0)
                xx = x0
                while xx < x1:
                    xb = min(x1, xx + 50.0)
                    near = abs((zz + zb) / 2 - eye[2]) < 60 and abs((xx + xb) / 2 - eye[0]) < 60
                    if near:      # split the tiles round the camera so none cross its near plane
                        for sz_ in np.arange(zz, zb, 2.0):
                            for sx_ in np.arange(xx, xb, 2.0):
                                tiles.append((sx_, min(xb, sx_ + 2.0), sz_, min(zb, sz_ + 2.0), kind, x1 - x0 < z1 - z0))
                    else:
                        tiles.append((xx, xb, zz, zb, kind, x1 - x0 < z1 - z0))
                    xx = xb
                zz = zb
        for xx, xb, zz, zb, kind, along_z in tiles:
            col = G.hx('#11161e') if kind == 'road' else G.hx('#1a1e26')
            if True:
                if True:
                    q = [(xx, 0.0, zb), (xb, 0.0, zb), (xb, 0.0, zz), (xx, 0.0, zz)]
                    if visible(q):
                        dist = np.linalg.norm(np.mean(q, 0) - eye)
                        ppm = float(np.clip(cam.f * ss / max(dist, 1), 0.3, 8))
                        tw, th = int(max(2, (xb - xx) * ppm)), int(max(2, (zb - zz) * ppm))
                        t = np.empty((th, tw, 3), np.float32)
                        t[:] = col
                        e = np.zeros_like(t)
                        if ppm >= 2 and th > 8 and tw > 8:   # asphalt: sponge texture, stains, puddles, markings
                            t *= (1 + 0.14 * G.field(rs, th, tw, max(2, ppm), max(2, ppm)))[..., None]
                            G.stains(t, rs, max(1, th * tw // 3000), '#05060a', size=(ppm * 0.5, ppm * 3), alpha=(0.2, 0.5))
                            for _ in range(int(rs.integers(0, 3))):
                                px_, py_ = rs.uniform(0, tw), rs.uniform(0, th)
                                G.dab(t, rs, px_, py_, px_ + rs.uniform(1, 4) * ppm, py_ + rs.uniform(0.5, 2) * ppm,
                                      G.hx('#2a3448'), rough=0.9)
                            if kind == 'road' and along_z and ppm >= 3 and (xb - xx) > 6:
                                for k_ in range(0, th, int(6 * ppm)):
                                    G.dab(t, rs, tw * 0.5 - 0.08 * ppm, k_, tw * 0.5 + 0.08 * ppm, k_ + 3 * ppm,
                                          G.hx('#9a9a8a'), alpha=0.7, rough=0.8)
                            if kind == 'walk':
                                for _ in range(int(rs.integers(0, 5))):
                                    dx_, dy_ = rs.uniform(0, tw), rs.uniform(0, th)
                                    G.dab(t, rs, dx_, dy_, dx_ + rs.uniform(0.2, 0.7) * ppm, dy_ + rs.uniform(0.1, 0.4) * ppm,
                                          ['#c8c4b4', '#8a3a30', '#6a7080'][rs.integers(3)], rough=0.9)
                        if kind == 'road' and eye[1] > 40:      # a river of car lights from above [5]
                            for lane, lc in ((0.3, '#fff0d0'), (0.7, '#ff4a30')):
                                for _ in range(int(rs.integers(1, 4))):
                                    L = max(1, int(rs.uniform(4, 15) * ppm))
                                    if along_z:
                                        x = int(tw * lane)
                                        ya = int(rs.uniform(0, th))
                                        t[ya:ya + L, max(0, x - 1):x + 1] = G.hx(lc)
                                        e[ya:ya + L, max(0, x - 1):x + 1] = G.hx(lc) * 0.8
                                    else:
                                        y = int(th * lane)
                                        xa = int(rs.uniform(0, tw))
                                        t[max(0, y - 1):y + 1, xa:xa + L] = G.hx(lc)
                                        e[max(0, y - 1):y + 1, xa:xa + L] = G.hx(lc) * 0.8
                        # continuous haze across the tile (no seams): per-texel distance
                        gx = np.linspace(xx, xb, tw, dtype=np.float32)[None, :]
                        gz = np.linspace(zb, zz, th, dtype=np.float32)[:, None]
                        dd = np.sqrt((gx - eye[0]) ** 2 + (gz - eye[2]) ** 2 + eye[1] ** 2)
                        a_ = np.clip((dd - 150) / 2400, 0, 0.55)[..., None]
                        t = t * (1 - a_) + G.hx('#3a2e40') * a_
                        st.face(q, t, emit=e, lit=False, order=-1e6)

    def _expressways(self, st, cam, ss, visible, hazed):
        eye = cam.eye
        teal = G.hx('#1f5a55')
        for d in self.decks:
            x, w, y = d['x'], d['w'], d['y']
            zz = d['z0']
            while zz < d['z1']:
                zb = min(d['z1'], zz + (4 if abs(zz - eye[2]) < 40 else 60))
                dist = np.linalg.norm(np.array([x, y, (zz + zb) / 2]) - eye)
                ppm = float(np.clip(cam.f * ss / max(dist, 1), 0.3, 10))
                top = [(x - w / 2, y, zb), (x + w / 2, y, zb), (x + w / 2, y, zz), (x - w / 2, y, zz)]
                if eye[1] > y and visible(top):
                    t = np.empty((max(2, int((zb - zz) * ppm)), max(2, int(w * ppm)), 3), np.float32)
                    t[:] = G.hx('#1c2028')
                    for lx in (0.33, 0.66):
                        xi = int(t.shape[1] * lx)
                        for k in range(0, t.shape[0], max(2, int(6 * ppm))):
                            t[k:k + max(1, int(3 * ppm)), xi:xi + max(1, int(0.2 * ppm))] = G.hx('#8a8e96')
                    t, _ = hazed(t, np.zeros_like(t), dist, y)
                    st.face(top, t, lit=True)
                for sx in (x - w / 2, x + w / 2):
                    s = [(sx, y + 1.0, zz), (sx, y + 1.0, zb), (sx, y - 1.4, zb), (sx, y - 1.4, zz)]
                    if eye[0] > sx:
                        s = [s[1], s[0], s[3], s[2]]
                    if visible(s):
                        t = np.empty((max(2, int(2.4 * ppm)), max(2, int(60 * ppm)), 3), np.float32)
                        t[:] = G.hx('#3a4048')
                        t[:max(1, t.shape[0] // 3)] = G.hx('#6a7078')
                        t, _ = hazed(t, np.zeros_like(t), dist, y)
                        st.face(s, t, lit=True)
                pz = zz + 30
                pier = [(x - 1.4, y - 1.4, pz), (x + 1.4, y - 1.4, pz), (x + 1.4, 0, pz), (x - 1.4, 0, pz)]
                if visible(pier):
                    t = np.empty((max(2, int(y * ppm)), max(2, int(2.8 * ppm)), 3), np.float32)
                    t[:] = teal
                    t[:, :max(1, t.shape[1] // 4)] *= 1.3
                    t, _ = hazed(t, np.zeros_like(t), dist, 0)
                    st.face(pier, t, lit=True)
                    cap = [(x - w / 2 + 1, y - 1.4, pz), (x + w / 2 - 1, y - 1.4, pz), (x + w / 2 - 1, y - 2.6, pz),
                           (x - w / 2 + 1, y - 2.6, pz)]
                    t2 = np.empty((max(2, int(1.2 * ppm)), max(2, int(w * ppm)), 3), np.float32)
                    t2[:] = teal * 0.85
                    st.face(cap, t2, lit=True)
                zz = zb
