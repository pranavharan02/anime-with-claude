# anime with claude

Hand-drawn-looking anime made entirely in code. No image models, no 3D, no
traced footage: every line, cel and painted background is placed by code, then
"shot on film" by a code camera.

The first phase builds style references by studying a film and reproducing its
look in original compositions. Study 1 is **Akira (1988)**.

![Akira study, plates 1-4](plates/akira/contact_sheet.png)

## How a frame is made

Each frame follows the order a 1988 cel production used:

1. **Background painting** (`engine/city.py`, `engine/paint.py`): poster-colour
   towers in depth layers, gradients toward the street glow, window grids lit in
   clusters, haze between layers, then pigment texture (brush streaks, tooth,
   mottling).
2. **Cels** (`engine/ink.py`): shapes authored as control points on
   centripetal Catmull-Rom curves, so the code works like placing pencil marks.
   Each material gets a flat base, one hard-edged shadow tone and sometimes a
   highlight. Lines are traced strokes with taper, slight width variation and
   hand wobble. Hair is inked as the union silhouette of its locks.
3. **Transmitted light** (`engine/fx.py`): lamps, windows and the tail-light
   trail are drawn into a separate emission pass and bloomed, the way backlit
   light was photographed through the cels.
4. **Film** (`engine/film.py`): gate weave, halation, lens softness, a highlight
   shoulder, lifted cool blacks, vignette, 35 mm grain and dust.

Rendering uses [skia-python](https://github.com/kyamagu/skia-python) for
anti-aliased vectors and NumPy/OpenCV for compositing and film effects.

## Study 1: Akira

`refs/akira/STYLE.md` records what makes a frame read as Akira (palette, cel
rules, background rules, light, film). The plates are original compositions in
that style, not copies of frames from the film.

| Plate | What it tests |
| --- | --- |
| `p01_city` | Background painting: depth layers, facades, haze, searchlights |
| `p02_trail` | Bike and rider cel, tail-light ribbon, parallax speed background |
| `p03_rider` | Character close-up: clumped hair, two-tone night shading, rim light |
| `p04_dome` | Effects: the white dome, shock rings, rim-lit silhouettes, light rays |
| `m01_trail` | Motion: 3 s at 24 fps, cel on twos, trail from the tail light's real path |

Finished plates are in `plates/akira/`.

## Run it

```bash
pip install -r requirements.txt
python studies/akira/p01_city.py
python studies/akira/p02_trail.py            # still
python studies/akira/p02_trail.py --motion   # 72 frames + MP4 (needs ffmpeg)
python studies/akira/p03_rider.py
python studies/akira/p04_dome.py
```

Output goes to `out/` (gitignored).

## Where study 1 falls short of the film

- **Faces:** the close-up reads as anime, but closer to modern TV animation
  than Otomo. It needs heavier, more realistic structure (cheekbones, eye
  bags, lip shapes) and less symmetric, rounded forms.
- **Bike:** the shell is too bulbous and smooth. The film's bike is longer,
  lower and more mechanical, with more panel detail.
- **Backgrounds:** the facades are still too regular. Mizutani's paintings have
  more structural detail (setbacks, signage, pipes, catwalks) and stronger
  value contrast between near and far.
- **Dome plate:** the silhouettes are plain rectangles. They need rooftop
  detail and debris.
