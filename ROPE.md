# The rotary position encoding in `romae_lc`

This note is for someone who wants to use or test the rotary encoding on
their own data. It says what the code does, where it is, and how to run it.

## What it is

Rotary position encoding (RoPE) tells attention where each token is. It
rotates the query and the key of every token by an angle that depends on the
token's position. The attention score between two tokens then depends only on
the distance between them.

Standard RoPE is made for text, where positions are 0, 1, 2, and so on. This
code works with real-valued positions, such as observation times in days. It
also works with more than one position axis, such as time and wavelength.

## What is new here

| | standard RoPE | this code |
|---|---|---|
| Positions | whole numbers | any real number |
| Timescales | one fixed formula | any list you give |
| Heads | all heads share one list | each head can have its own list |
| Layers | all layers share one rotation | each layer can have its own |
| Axes | one | several, each with its own block of channels |

With one list per head and per layer, a small model can cover hundreds of
different timescales. Our light model has 6 layers, 3 heads, and 21 angles per
head. That gives 378 timescales. With one shared list it had 12.

## Where the code is

| file | what is in it |
|---|---|
| `romae_lc/rope.py` | the encoding. Start here. |
| `romae_lc/transformer.py` | where the rotation is applied, inside attention |
| `romae_lc/model.py` | how the encoder builds one rotation per layer |
| `tests/test_rope.py` | the tests |

The main names in `romae_lc/rope.py`:

| name | what it does |
|---|---|
| `AxialRope` | the rotation for one axis. Takes a list of timescales. A nested list gives every head its own. |
| `SimplexRope` | one rotation for a group of axes together |
| `BlockRope` | joins the blocks of one head, for example one for time and one for band |
| `Rotation` | the sin and cos tables for one batch. Call it on queries and keys. |
| `layout` | builds the standard split of a head into blocks |
| `collapse_layout` | folds many per-head lists into one shared list, for a decoder |
| `resample_ladder` | changes a list of timescales to a new length |

## Three words to know

**Wavelength.** The distance in position after which one channel pair has made
a full turn. If your positions are in days, the wavelength is in days. This is
not the wavelength of light.

**Timescale.** The wavelength divided by 2π. This is the number the code takes
as input.

```
angle = position / timescale
wavelength = 2π × timescale
```

**Ladder.** The list of timescales. One entry of it is one rung.

## Quick start 1: use it in your own attention

This needs only `rope.py`. It was run and checked.

```python
import numpy as np, torch
from romae_lc.rope import BlockRope

B, N, H, D = 2, 50, 4, 16        # batch, tokens, heads, channels per head

# 6 wavelengths for each of the 4 heads, in the units of your positions
wavelengths = np.geomspace(0.1, 100, H * 6).reshape(H, 6)

layout = [dict(kind="axial", axes=[0], dim=D,
               timescales=(wavelengths / (2 * np.pi)).tolist())]
rope = BlockRope(head_dim=D, nhead=H, layout=layout)

t = torch.rand(B, N) * 30                 # the position of every token
rot = rope.prepare(t[:, None, :])         # positions have shape [B, axes, N]

q = torch.randn(B, N, H, D)               # queries, before the heads are moved to axis 1
k = torch.randn(B, N, H, D)
q, k = rot(q), rot(k)                     # rotate, then do attention as usual
```

Rotate the queries and the keys. Do not rotate the values.

## Quick start 2: a full encoder with one ladder per layer and per head

```python
import numpy as np, torch
from romae_lc import RoMAE
from romae_lc.rope import layout

depth, heads, width = 2, 2, 24            # channels per head: 24 / 2 = 12
blocks = layout(12, 2, p=0.75, time_frac=0.875)   # a time block and a band block
n = 3                                     # active angles in the time block

# one wavelength per angle, per head, per layer
w = np.geomspace(0.05, 500, depth * heads * n).reshape(depth, heads, n)

layers = []
for l in range(depth):
    lay = [dict(b) for b in blocks]
    lay[0]["timescales"] = (w[l] / (2 * np.pi)).tolist()
    layers.append(lay)

enc = RoMAE(encoder=dict(d_model=width, nhead=heads, depth=depth),
            n_channels=1, n_axes=2, rope=layers)

values = torch.randn(3, 10, 1)                          # [B, N, channels]
positions = torch.stack([torch.rand(3, 10) * 100,       # axis 0: time
                         torch.full((3, 10), 480.0)], 1)    # axis 1: band
print(enc(values, positions).shape)                     # torch.Size([3, 24])
```

`enc.rope_layout` gives back the full layout as plain lists. Save it with the
checkpoint. Pass it as `rope=` to build the same encoder again.

## How the channels of a head are used

One head has `head_dim` channels. They are split into blocks. Each block
belongs to one axis, or to a group of axes.

Inside a block with `dim` channels there are `dim / 2` angles. Each angle
rotates one pair of channels.

Not every angle has to be active. An angle that is not active does not rotate.
Those channels carry content with no position in it. By default 75 percent of
the angles are active (`p = 0.75`).

If you give your own list of timescales, its length sets the number of active
angles. The list can hold at most `dim / 2` values.

Example from our light model:

| | value |
|---|---|
| Channels per head | 64 |
| Channels for time | 56 |
| Channels for band | 8 |
| Angles in the time block | 28 |
| Active angles | 21 |

## How to choose the timescales for your data

1. **Find the range.** The shortest wavelength should be near the shortest
   pattern you care about. The longest should be about twice the length of
   your input. Then the slowest channel makes half a turn over the input.
2. **Pick the spacing.** Log spacing is a safe start. You can also put more
   rungs where your data has more patterns.
3. **Count the rungs.** You have `layers × heads × active angles` of them.
4. **Hand them out.** Give each head a group of rungs that lie close together.
   Let each layer cover the whole range.

For light curves, the code that does this is in `project/common.py`:
`measure_wavelengths`, `deal_ladder`, `dense_ladder`, `rope_layouts` and
`build_backbone`. It reads the patterns from the data. For other data you can
skip it and pass your own lists, as in the examples above.

## How close must a rung be to a period

A channel with wavelength λ lines up points that are a whole number of λ
apart. If the true period is P and the rung is off by a small share ε, the
error grows with every cycle. Over N cycles it stays small only if:

```
ε < 1 / (4 × N)
```

So an input with 100 cycles needs a rung within 0.25 percent of the period.

In practice the model did better than this rule says. The score is a sum over
many rungs, and rungs that lie close together work as a group. Treat the rule
as a safe limit, not a hard one.

## A decoder with a different number of heads

A decoder may have fewer heads than the encoder. It cannot take the per-head
lists as they are. `collapse_layout` joins all the lists into one and brings
it to the right length:

```python
from romae_lc.rope import collapse_layout
shared = collapse_layout(enc.rope_layout)     # one layout, one list
```

## Things to watch

- **Keep positions small.** The angle is `position / timescale`. With a large
  position and a short wavelength the angle gets very large, and 32-bit
  numbers lose detail. In our check, the scores of shifted positions agreed to
  1 part in a trillion with 64-bit numbers, but only to 1 part in 2,000 with
  32-bit ones. Set the start of each input to zero.
- **Without a CLS token the model cannot know absolute position.** It only
  sees distances. Add a CLS token at position 0 if absolute position matters.
- **`head_dim` must be even.** So must the `dim` of every block.
- **The `dim` values of the blocks must add up to `head_dim`.**
- **A nested list must have one row per head.** All rows must have the same
  length.
- **Timescales must be positive and finite.**
- **The cost.** Positions change with every batch, so the sin and cos tables
  are built again for each one. The RoMAE paper measured about 13 percent
  slower than fixed positions.

## Run the tests

```
.venv/bin/python -m pytest tests/test_rope.py -q
```

The tests for the new parts:

| test | what it checks |
|---|---|
| `test_per_head_ladders` | each head gets its own list |
| `test_per_head_rotation_is_relative_and_the_layout_serialises` | the score depends only on distance, and the layout can be saved and loaded |
| `test_axial_explicit_timescales_in_any_spacing` | any list of timescales works |
| `test_collapse_layout` | many lists fold into one |
| `test_nope_channels_are_not_rotated` | the angles that are not active do nothing |

## What we saw on light curves

These are results from our own project. They may not hold for other data.

- With the old training goal, going from 12 rungs to 378 changed nothing. The
  training goal was the limit, not the encoding.
- With masked training, the light model (378 rungs) and the wide model (756
  rungs) both learned the period. The wide one was a little better for the two
  largest classes. The R2 of a straight-line fit of log period from the latent
  was 0.85 against 0.81 for eclipsing binaries, and 0.74 against 0.69 for RR
  Lyrae.
- Both stopped getting better on the period after 10,000 to 20,000 steps.
  More data would help more than more rungs.

## References

- Zivanovic and others, 2025. Rotary Masked Autoencoders are Versatile
  Learners. arXiv:2505.20535.
- Su and others, 2021. RoFormer: Enhanced Transformer with Rotary Position
  Embedding.
- Barbero and others, 2025. Round and Round We Go! What makes Rotary
  Positional Encodings useful? (the source of `p`, the active share)
