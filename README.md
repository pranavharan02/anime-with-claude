# anime with claude

Can anime frames that look hand-painted be made entirely in code, with no image
models, no 3D packages, and no traced footage? This repository is a working log
of trying, built in collaboration with Claude. It contains a cel-paint engine, a
tool that turns a still into editable scene code, a set of original frames, and
an original procedurally generated megacity in the style of *Akira* (1988).
Blind AI critics score every result.

![Four camera setups on one generated city: a street canyon, a worm's-eye view, a bird's-eye view, and a telephoto skyline at night](plates/city/contact_sheet.png)

## Results at a glance

The short version: reconstructing an existing frame as code works well, but
original frames don't yet pass as the real thing.

| Experiment | What it does | Result |
| --- | --- | --- |
| [Frame to code](#frame-to-code-vectorizer) | Rebuilds a still as layered scene code (airbrush mesh, cel paths, ink strokes) | Mean color error ΔE 0.7 to 2.0, below the threshold most people notice |
| [Style plates](#style-plates) | Original compositions drawn from hand-placed curves | Read as "anime", not as the target film |
| [Original frames](#original-frames) | New frames in the same scene format, then a painted 2.5D stage | 2% to 8% judged genuine |
| [Original city](#an-original-megacity) | A whole city, generated once and shot from many cameras | 3% to 6% judged genuine, and one skyline at 22% |

Real frames from the film, in the same blind rounds, scored 80% to 98%.

## How it's measured

The blind test in `tools/blind.py` shuffles candidate frames among real frames
from the reference film. It resizes every image to the same size and JPEG
quality, crops letterbox bars, and writes a hidden answer key. A fresh judge
agent, with no knowledge of the project, then sees only that folder. The judge
rates each frame's chance of being genuine and lists the tells. Later rounds
ran two independent judges in parallel.

The judge also turns out to be harsh on real frames: across rounds, genuine
frames ranged from 30% to 98%. Keep that in mind when you read the scores.

## The experiments

### Style plates

Original compositions built from scratch: a night city, a bike with a
tail-light trail, a character close-up, and an energy dome. Shapes are
authored as control points on centripetal Catmull-Rom curves, inked as tapered
strokes with slight hand wobble, and shot through a simulated 35 mm print
(gate weave, halation, grain, and dust).

- Code: `engine/core.py`, `engine/ink.py`, `engine/film.py`, `studies/akira/`
- Plates: `plates/akira/`, including a 3-second motion test (`m01_trail.mp4`)

![Four style plates: a night city, a bike with a light trail, a rider close-up, and a white dome](plates/akira/contact_sheet.png)

### Frame-to-code vectorizer

`engine/trace.py` reads a reference still and writes it back out as a JSON
scene with three layers, in the order a cel was built:

- **Airbrush:** a gradient mesh of triangles with per-corner colors.
- **Paint:** cel regions as sub-pixel paths, each with a measured edge
  feather.
- **Ink:** line centerlines with per-point width and color.

The pipeline finds ink with a morphological `black-hat` filter and traces its
skeleton. It then inpaints the ink away, finds the paint palette with k-means
in Lab color space, and traces each region with marching squares. Last, it
runs analysis by synthesis: it renders the scene, measures the error inside
every shape, corrects that shape's color, and repeats.
`engine/scene.py` renders the result with Skia.

On five test stills the reconstruction reached a mean CIEDE2000 color error of
0.7 to 2.0 and SSIM of 0.90 to 0.96. At 2x zoom, fine ink strokes run thin, and
JPEG artifacts in the source get traced as paint.

No reference frames or reconstructions are included in this repository. To try
the tool, supply your own stills; see [Get started](#get-started).

### Original frames

New frames written directly in the scene format, with defaults measured from
the reference material: ink color and width, palette, lens softness, and
grain. Later frames use a 2.5D painting stage (`engine/stage.py`): every
surface is a quad in 3D, painted flat as a gouache texture
(`engine/gouache.py`), lit per texel, and warped into perspective.

![An original night street with signs, a vending machine, wet reflections, and a red tail-light trail](plates/akira_originals/o03_street.png)

### An original megacity

`engine/citygen.py` generates a whole city once, about 38,000 building parts,
and `studies/city/shots.py` films it from five camera setups. It follows a
style rule book, `refs/akira/CITY_BIBLE.md`, which an analysis agent wrote
from about 1,500 sampled film frames. The city has:

- Three building strata: grimy low-rise, office slabs, and a core of
  megatowers built from stacked prisms with setbacks, belts, crowns, and spires.
- A hue family per tower, with warm and cool towers alternating.
- Facades painted per shot at their distance class (`engine/facade.py`), with
  gouache brush tiles (`engine/brushtex.py`) and hand-placed windows.
- Haze in discrete planes that glows from the street, expressways on teal
  piers, rooftop plant, billboards with invented brand names, and inked cel
  traffic.

Nothing in the city copies a building or character from any film.

## Lessons from the critics

Across five city rounds and two judges per round, the same tells came back:

- **Computed perspective.** A mathematically exact vanishing point reads as a
  3D render. Telephoto skylines, with almost no convergence, scored highest.
- **Regular window lattices.** Even with jittered sizes and colors, an
  underlying grid shows through.
- **Even light.** Real night backgrounds are mostly near-black, with light
  pooling and bleeding from a few sources.
- **Procedural texture.** Noise reads as noise. A uniform grain overlay made
  scores worse. A brush-stroke repaint pass smeared edges and also made
  them worse.
- **Recognition.** Judges recognize scenes and characters from the real film,
  which gives genuine frames a boost no original frame can earn.

The most promising untried direction is a multiplane approach: layered 2D
painted flats that slide at different speeds, instead of a 3D camera.

## Get started

You need Python 3.11 or later. `ffmpeg` is optional and only used for MP4
output.

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

1. Render the style plates:

   ```bash
   python studies/akira/p01_city.py
   python studies/akira/p02_trail.py --motion
   ```

1. Generate the city and render its shots:

   ```bash
   python studies/city/shots.py canyon skyline
   ```

   Renders go to `out/`, which is gitignored.

Some tools work on reference frames that you supply. They read from folders
that are gitignored, so nothing you add is committed:

| Tool | Reads |
| --- | --- |
| `studies/akira_frames/recreate.py` | Stills in `ref/akira/` named `akira_NNN.jpg` |
| `tools/texbank.py` | Frames in `ref/akira_film/` named `f_NNNNN.jpg`; writes `out/texbank.npz` |
| `tools/blind.py` | Real frames from `ref/akira_film/` to shuffle among your candidates |

Without a texture bank, the print stage falls back to synthetic grain. Sign
lettering looks for a font with Japanese glyphs and falls back to a default
font if none is installed.

## Repository layout

| Path | Contents |
| --- | --- |
| `engine/` | Rendering engine: ink, film print, scene format, vectorizer, 2.5D stage, gouache painting, city generator |
| `studies/` | One folder per experiment: style plates, frame recreation, original frames, the city |
| `tools/` | Blind-test harness and texture-bank builder |
| `refs/akira/` | Written style analysis: `STYLE.md` and `CITY_BIBLE.md` |
| `plates/` | Finished renders, all original work |

## A note on copyright

*Akira* is the work of Katsuhiro Otomo and its rights holders. This repository
contains no frames, stills, or footage from the film, and no reconstructions
of them. The written analysis in `refs/` is commentary on style. Every image
in `plates/` is an original composition made by the code here.
