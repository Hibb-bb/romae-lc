"""Token-level JEPA pretraining and evaluation for light curves, built on ``romae_lc``.

:mod:`project.train` trains the encoder (one recipe file per run in
``project/recipes/``). ``project/eval/`` scores it: :mod:`project.eval.encode` caches the
window latents, the ``card_*`` scripts are the four test cards, and
:mod:`project.eval.test_card` runs them all for one run. Older code is in ``legacy/``. Run every script as
a module from the repository root, e.g.
``python -m project.train --help``.
"""
