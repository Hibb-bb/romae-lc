"""Latent world model for light curves with energy-based inference.

An extension of ``romae_lc`` following ``lc-world-model-design.md``: stage 1
trains the package's :class:`~romae_lc.LeWorldModel` on ZTF windows with the
per-point error as a second token channel (:mod:`project.train_wm`); stage 2
fits the residual of the deterministic predictor (:mod:`project.residual`);
stage 3 trains a decoder from the frozen latent back to clean magnitudes
(:mod:`project.decoder`); :mod:`project.energy` composes the pieces into
energies for smoothing, forecasting, anomaly scoring and period detection
(:mod:`project.infer`), and :mod:`project.inject` is the injected-anomaly test.
Run every script as a module from the repository root, e.g.
``python -m project.train_wm --help``.
"""
