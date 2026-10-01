"""Scene code: save, load and render the layered cel scenes from trace.py.

A scene file is plain JSON:
  {"w", "h",
   "mesh":  {"step", "cols", "rows", "colors": ["#rrggbb", ...]},   airbrush
   "fills": [[fill, [x0, y0, x1, y1, ...]], ...],                   paint
            optional 3rd item: feather (edge blur sigma, px) for airbrushed edges
            fill = "#rrggbb" | ["lin", x0, y0, x1, y1, "#c0", "#c1"]
   "lines": [["#rrggbb", [x, y, w, x, y, w, ...]], ...]}           ink
"""
import json

import numpy as np
import skia


def _hex(c):
    c = np.clip(np.round(c), 0, 255).astype(int)
    return '#%02x%02x%02x' % tuple(c)


def _rgb(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _r(v):
    return [round(float(a), 1) for a in v]


def to_json(sc):
    m = sc['mesh']
    out = dict(w=sc['w'], h=sc['h'],
               mesh=dict(step=m['step'], cols=m['cols'], rows=m['rows'],
                         colors=[_hex(c) for c in m['colors'].reshape(-1, 3)]),
               fills=[], lines=[])
    for _, fill, pts, feather, *_ in sc['fills']:
        f = _hex(fill[1]) if fill[0] == 'flat' else ['lin', *_r(fill[1]), *_r(fill[2]), _hex(fill[3]), _hex(fill[4])]
        item = [f, _r(pts.reshape(-1))]
        if feather > 0:
            item.append(round(feather, 1))
        out['fills'].append(item)
    for pts, wid, colr, *_ in sc['lines']:
        out['lines'].append([_hex(colr), _r(np.column_stack([pts, wid]).reshape(-1))])
    return json.dumps(out, separators=(',', ':'))


def _color(h, a=255):
    r, g, b = _rgb(h)
    return skia.Color(r, g, b, a)


def render(scene, scale=1.0, layers=('mesh', 'fills', 'lines')):
    """Scene dict (as loaded from JSON) -> float RGB 0..1."""
    w, h = scene['w'], scene['h']
    W, H = int(round(w * scale)), int(round(h * scale))
    info = skia.ImageInfo.Make(W, H, skia.kRGBA_8888_ColorType, skia.kPremul_AlphaType)
    surf = skia.Surface.MakeRaster(info)
    c = surf.getCanvas()
    c.clear(skia.ColorBLACK)
    c.scale(scale, scale)

    if 'mesh' in layers:
        m = scene['mesh']
        cols, rows, st = m['cols'], m['rows'], m['step']
        pos = [skia.Point(min(i * st, w), min(j * st, h)) for j in range(rows) for i in range(cols)]
        clr = [_color(x) for x in m['colors']]
        idx = []
        for j in range(rows - 1):
            for i in range(cols - 1):
                a = j * cols + i
                idx += [a, a + 1, a + cols, a + 1, a + cols + 1, a + cols]
        v = skia.Vertices(skia.Vertices.kTriangles_VertexMode, pos, None, clr, idx)
        c.drawVertices(v, skia.Paint(AntiAlias=True), skia.BlendMode.kDst)

    if 'fills' in layers:
        for item in scene['fills']:
            fill, flat = item[0], item[1]
            feather = item[2] if len(item) > 2 else 0
            p = skia.Path()
            p.moveTo(flat[0], flat[1])
            for i in range(2, len(flat), 2):
                p.lineTo(flat[i], flat[i + 1])
            p.close()
            paint = skia.Paint(AntiAlias=True)
            if isinstance(fill, str):
                paint.setColor(_color(fill))
            else:
                _, x0, y0, x1, y1, c0, c1 = fill
                paint.setShader(skia.GradientShader.MakeLinear(
                    [skia.Point(x0, y0), skia.Point(x1, y1)], [_color(c0), _color(c1)]))
            if feather:
                # airbrush: paint sprayed through a soft-edged mask
                paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, feather))
                c.drawPath(p, paint)
                continue
            c.drawPath(p, paint)
            paint.setStyle(skia.Paint.kStroke_Style)  # close the anti-aliased seams
            paint.setStrokeWidth(1.0)
            paint.setStrokeJoin(skia.Paint.kRound_Join)
            c.drawPath(p, paint)

    if 'lines' in layers:
        for colr, flat in scene['lines']:
            a = np.array(flat).reshape(-1, 3)
            paint = skia.Paint(AntiAlias=True, Color=_color(colr), Style=skia.Paint.kStroke_Style,
                               StrokeCap=skia.Paint.kRound_Cap)
            if len(a) == 1:
                c.drawCircle(a[0, 0], a[0, 1], a[0, 2] / 2, skia.Paint(AntiAlias=True, Color=_color(colr)))
                continue
            for i in range(len(a) - 1):
                paint.setStrokeWidth(float((a[i, 2] + a[i + 1, 2]) / 2))
                c.drawLine(a[i, 0], a[i, 1], a[i + 1, 0], a[i + 1, 1], paint)
    return surf.makeImageSnapshot().toarray()[..., :3].astype(np.float32) / 255.0


def refine(scene, pix, ref, iters=4, lens=0.0, rate=1.0):
    """Analysis by synthesis: render the code, measure the error inside each
    shape, correct that shape's colour, repeat. `pix` holds, per fill then per
    line, the (ys, xs) pixels the shape is responsible for."""
    import cv2
    target = ref.astype(np.float32)
    nf = len(scene['fills'])
    for _ in range(iters):
        out = render(scene)
        if lens:
            out = cv2.GaussianBlur(out, (0, 0), lens)
        res = target - out * 255.0
        for k, item in enumerate(scene['fills']):
            ys, xs = pix[k]
            d = res[ys, xs].mean(0) * rate
            f = item[0]
            if isinstance(f, str):
                item[0] = _hex(np.array(_rgb(f)) + d)
            else:
                f[5] = _hex(np.array(_rgb(f[5])) + d)
                f[6] = _hex(np.array(_rgb(f[6])) + d)
        for j, item in enumerate(scene['lines']):
            ys, xs = pix[nf + j]
            d = res[ys, xs].mean(0) * rate
            item[0] = _hex(np.array(_rgb(item[0])) + d)
    return scene
