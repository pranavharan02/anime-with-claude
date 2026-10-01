"""2.5D background stage: painted textures on 3D quads, seen through a camera.

A background painter works in perspective but paints flat surfaces. So:
each surface is a quad in world space (metres) with a texture painted flat in
its own space (gouache tools in gouache.py), lit by point lights evaluated per
texel in 3D, then warped into the frame with the quad's homography. Painter's
order by depth. Rendered supersampled, then reduced, which anti-aliases edges.

World: x right, y up, z forward (into the picture).
"""
import cv2
import numpy as np


class Camera:
    def __init__(self, w, h, fov_deg=55, eye=(0, 1.5, 0), yaw=0.0, pitch=0.0):
        self.w, self.h = w, h
        self.f = (w / 2) / np.tan(np.radians(fov_deg) / 2)
        self.eye = np.array(eye, float)
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        self.R = rx @ ry  # world -> camera

    def to_cam(self, P):
        return (np.asarray(P, float) - self.eye) @ self.R.T

    def project(self, P):
        c = self.to_cam(P)
        z = np.maximum(c[..., 2], 1e-3)
        x = self.w / 2 + self.f * c[..., 0] / z
        y = self.h / 2 - self.f * c[..., 1] / z
        return np.stack([x, y], -1), c[..., 2]


class Stage:
    def __init__(self, cam, ambient=(0.30, 0.32, 0.45), fog=None):
        """fog = (colour rgb 0..1, distance in metres at which ~63% is haze)."""
        self.cam = cam
        self.faces = []
        self.lights = []
        self.ambient = np.array(ambient, np.float32)
        self.fog = fog

    def light(self, pos, color, power=1.0, radius=4.0):
        self.lights.append((np.array(pos, float), np.array(color, np.float32), power, radius))

    def face(self, corners, tex, emit=None, alpha=None, lit=True, order=0.0, outline=None, line_w=1.1):
        """corners: 4 world points in texture order (top-left, top-right,
        bottom-right, bottom-left). tex: (th, tw, 3) albedo 0..1."""
        self.faces.append(dict(c=np.array(corners, float), tex=tex, emit=emit, alpha=alpha, lit=lit,
                               order=order, outline=outline, line_w=line_w))

    def box(self, x0, x1, y0, y1, z0, z1, tex_front, tex_side=None, tex_top=None, **kw):
        """A box against the scene: the faces a camera at the eye can see."""
        ex = self.cam.eye
        self.face([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], tex_front, **kw)   # front (z0)
        if tex_side is not None:
            if ex[0] > x1:
                self.face([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], tex_side, **kw)
            elif ex[0] < x0:
                self.face([(x0, y1, z1), (x0, y1, z0), (x0, y0, z0), (x0, y0, z1)], tex_side, **kw)
        if tex_top is not None and ex[1] > y1:
            self.face([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], tex_top, **kw)

    # -- lighting --------------------------------------------------------------
    def shade(self, fc):
        tex = self._light(fc)
        if self.fog is None:
            return tex
        col, dist = self.fog
        c = fc['c']
        th, tw = tex.shape[:2]
        if th * tw > 4:
            u = np.linspace(0, 1, tw, dtype=np.float32)[None, :, None]
            v = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
            top = c[0] + (c[1] - c[0]) * u
            bot = c[3] + (c[2] - c[3]) * u
            d = np.linalg.norm(top + (bot - top) * v - self.cam.eye, axis=-1, keepdims=True)
        else:
            d = np.full((th, tw, 1), np.linalg.norm(c.mean(0) - self.cam.eye), np.float32)
        f = (1 - np.exp(-d / dist)).astype(np.float32)
        fc['_fog'] = f
        return tex * (1 - f) + np.asarray(col, np.float32) * f

    def _light(self, fc):
        tex = fc['tex']
        if not fc['lit']:
            return tex
        th, tw = tex.shape[:2]
        c = fc['c']
        u = np.linspace(0, 1, tw, dtype=np.float32)[None, :, None]
        v = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
        top = c[0] + (c[1] - c[0]) * u
        bot = c[3] + (c[2] - c[3]) * u
        P = top + (bot - top) * v                       # (th, tw, 3) world positions
        n = np.cross(c[1] - c[0], c[3] - c[0])
        n /= np.linalg.norm(n) + 1e-9
        if np.dot(self.cam.eye - c.mean(0), n) < 0:
            n = -n
        light = np.broadcast_to(self.ambient, P.shape).copy()
        for pos, col, power, rad in self.lights:
            d = pos - P
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            lam = np.clip((d @ n)[..., None] / (dist + 1e-6), 0, 1) * 0.7 + 0.3
            light += col * power * lam / (1 + (dist / rad) ** 2)
        # a painter lays light in as a few values, not a smooth render ramp
        lv = light * 5.0
        fl = np.floor(lv)
        fr = lv - fl
        step = (fl + np.clip((fr - 0.4) / 0.2, 0, 1)) / 5.0
        light = 0.6 * step + 0.4 * light
        return tex * light

    # -- render ----------------------------------------------------------------
    def render(self, bg, ss=2):
        """bg: (h, w, 3) painted backdrop at output size. Returns (img, emit)."""
        h, w = bg.shape[:2]
        H, W = h * ss, w * ss
        img = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
        emit = np.zeros_like(img)
        ids = np.full((H, W), -1, np.int32)   # which surface owns each pixel (for paint-over)
        zb = np.full((H, W), 1e9, np.float32)  # depth of the surface that owns each pixel
        depth = lambda fc: -self.cam.to_cam(fc['c'])[:, 2].mean() + fc['order']
        for fi, fc in sorted(enumerate(self.faces), key=lambda t: depth(t[1])):
            cz = self.cam.to_cam(fc['c'])[:, 2]
            if (cz <= 0.05).any():
                continue
            dst, _ = self.cam.project(fc['c'])
            dst = (dst * ss).astype(np.float32)
            # only touch the face's own screen box: cost follows screen area
            x0, y0 = int(max(0, np.floor(dst[:, 0].min()) - 1)), int(max(0, np.floor(dst[:, 1].min()) - 1))
            x1, y1 = int(min(W, np.ceil(dst[:, 0].max()) + 2)), int(min(H, np.ceil(dst[:, 1].max()) + 2))
            if x1 - x0 < 1 or y1 - y0 < 1:
                continue
            bw, bh = x1 - x0, y1 - y0
            th, tw = fc['tex'].shape[:2]
            src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
            M = cv2.getPerspectiveTransform(src, dst - np.float32([x0, y0]))
            tex = self.shade(fc)
            a = fc['alpha'] if fc['alpha'] is not None else np.ones((th, tw), np.float32)
            wa = cv2.warpPerspective(a, M, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
            wt = cv2.warpPerspective(tex.astype(np.float32), M, (bw, bh), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            sub = img[y0:y1, x0:x1]
            sub[:] = sub * (1 - wa) + wt * wa
            if fc['outline']:
                oc = tuple(float(v) for v in _hex(fc['outline']))
                cv2.polylines(img, [np.round(dst * 16).astype(np.int32)], True, oc,
                              max(1, int(round(fc['line_w'] * ss))), cv2.LINE_AA, shift=4)
            own = wa[..., 0] > 0.5
            ids[y0:y1, x0:x1][own] = fi
            zb[y0:y1, x0:x1][own] = float(cz.mean())
            esub = emit[y0:y1, x0:x1]
            esub *= (1 - wa)
            if fc['emit'] is not None:
                em = fc['emit'].astype(np.float32)
                if '_fog' in fc:
                    em = em * (1 - 0.75 * fc['_fog'])
                esub += cv2.warpPerspective(em, M, (bw, bh),
                                            flags=cv2.INTER_LINEAR, borderValue=0) * wa
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        emit = cv2.resize(emit, (w, h), interpolation=cv2.INTER_AREA)
        self.ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)
        self.zbuf = cv2.resize(zb, (w, h), interpolation=cv2.INTER_NEAREST)
        return img, emit

    def stroke_dirs(self):
        """Brush direction per surface: vertical on walls, along the plane on floors."""
        dirs = {}
        for i, fc in enumerate(self.faces):
            c = fc['c']
            if np.allclose(c[:, 1], c[0, 1]):           # horizontal plane: stroke along its depth
                p0, _ = self.cam.project(c[0])
                p1, _ = self.cam.project(c[3])
                d = p1 - p0
                dirs[i] = float(np.arctan2(d[1], d[0]))
            else:
                dirs[i] = np.pi / 2
        return dirs


def _hex(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def quad_x(x, z0, z1, y0, y1):
    """A wall in the plane x = const, spanning z0..z1 and y0..y1 (seen from +x)."""
    return [(x, y1, z0), (x, y1, z1), (x, y0, z1), (x, y0, z0)]


def quad_z(z, x0, x1, y0, y1):
    """A wall facing the camera in the plane z = const."""
    return [(x0, y1, z), (x1, y1, z), (x1, y0, z), (x0, y0, z)]


def quad_y(y, x0, x1, z0, z1):
    """A floor or ceiling in the plane y = const (top-left is far-left)."""
    return [(x0, y, z1), (x1, y, z1), (x1, y, z0), (x0, y, z0)]
