# Akira (1988) style study: what we are matching

This is the reference sheet for the first style study. It records what makes a
frame read as *Akira* so the engine can reproduce the look in code. The plates
in `studies/akira/` are original compositions in this style, not copies of
frames from the film.

## The production, in one paragraph

Akira was painted by hand on 35 mm film at 24 fps, much of it animated on ones,
with 327 paint colours mixed for the film (around 50 of them only for night
scenes). Toshiharu Mizutani's backgrounds are poster-colour paintings with hard
edges and enormous detail. Characters are cels: thin traced lines, a flat base
colour, and one hard-edged shadow tone. Light sources (tail lights, neon,
screens) are photographed as transmitted light, so they glow and bloom on the
film instead of being painted.

## Visual rules

### Palette

- **Night is not black.** The darkest values are navy, teal-black, or aubergine.
  Pure `#000` only appears in line art and deep hair masses.
- **Night city:** teal-blue and cyan-green building faces, deep navy sky, a
  lighter blue-green haze at street level from the city glow.
- **Accent:** saturated red (the bike, the jacket) is the one hot colour in a
  cold frame. Under night light the red shifts toward crimson with a magenta
  shadow, but it stays the most saturated thing on screen.
- **Lights:** warm white-yellow cores with orange-red halos; some cyan and
  magenta neon.

### Characters (cels)

- Realistic proportions (Otomo design): smallish eyes with defined irises,
  heavy brows, a real nose with a cast shadow, wide mouths.
- Lines are thin (about 1.5-2.5 px at 1080p), nearly uniform, dark brown-black,
  with slight hand wobble and tapered open strokes.
- Two tones per material: base and one shadow, with a hard edge. Hair and red
  vinyl get a third tone (a highlight band). Shadows cool toward blue-violet at
  night.
- Night scenes repaint the characters in the night palette: skin goes
  grey-olive, red goes crimson, whites go blue-grey.

### Backgrounds (paintings)

- Many towers, layered in depth; far layers are lighter, bluer, and lower in
  contrast.
- Each face is flat colour with a vertical gradient (lighter toward the street
  glow) and painted edge highlights on lit corners.
- Windows are tiny rectangles in grids, mostly warm white, lit in clusters,
  never perfectly regular.
- Brush texture is subtle: slight streaking and uneven pigment, no visible
  digital gradients banding.

### Light and effects

- Tail light trails: a thick ribbon that persists in the air behind the bike,
  orange-red outer with a yellow-white core, heavily bloomed.
- Bloom on every light source; halation on the brightest highlights.
- Searchlight beams in the sky over the city.
- The explosion: a white sphere with a hard edge, shock rings, everything else
  flattened to silhouette.

### Camera and film

- 1.85:1 frame.
- Visible 35 mm grain, slight softness, a little gate weave, occasional dust.
- Pans slide the background under the cels; speed shots use horizontal
  speed-line backgrounds and motion-blurred lights.

## Plate list (study 1)

| Plate | What it tests |
| --- | --- |
| `p01_city` | Background painting: depth layers, window grids, haze, searchlights |
| `p02_trail` | Cel bike + rider, tail-light ribbon, speed background, bloom |
| `p03_rider` | Character close-up: face construction, two-tone shading, night light |
| `p04_dome` | Effects plate: white dome, shock rings, silhouette city |
| `m01_trail` | Motion: 3 s pan at 24 fps on twos, film gate weave and grain |
