"""Token-level JEPA pretraining and evaluation for light curves, built on ``romae_lc``.

:mod:`project.train` trains the encoder (one recipe file per run in
``project/recipes/``), :mod:`project.cache_latents`
caches its window latents, and the probes (:mod:`project.period_probe`,
``project/eval/``) score it. Older code is in ``legacy/``. Run every script as
a module from the repository root, e.g.
``python -m project.train --help``.
"""
