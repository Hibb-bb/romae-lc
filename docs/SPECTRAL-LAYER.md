# The spectral layer against Lomb-Scargle

A note before building. One window of a light curve has points $(t_j, y_j, \sigma_j, b_j)$, $j = 1..N$, at irregular times. The question is where a *learned* spectral layer inside the encoder is the same as the Lomb-Scargle periodogram, where it differs, and why the rotary attention we have cannot do the same job.

## 1. Lomb-Scargle

Lomb-Scargle fits one sinusoid at a trial frequency $f$ to the points by weighted least squares and reports how much of the variance the fit explains.

**Classic form (Scargle 1982).** With $y$ centred (mean removed) and unit weights,

$$
P(f) = \frac{1}{2}\left[\frac{\big(\sum_j y_j \cos\omega(t_j - \tau)\big)^2}{\sum_j \cos^2\omega(t_j - \tau)} + \frac{\big(\sum_j y_j \sin\omega(t_j - \tau)\big)^2}{\sum_j \sin^2\omega(t_j - \tau)}\right],
\qquad \omega = 2\pi f,
$$

where $\tau$ is chosen so the sine and cosine sums are orthogonal:
$\tan 2\omega\tau = \sum_j \sin 2\omega t_j / \sum_j \cos 2\omega t_j$.

**Generalised form (Zechmeister & Kürster 2009), the one in our code.** With weights $w_j \propto 1/\sigma_j^2$, $\sum_j w_j = 1$, a floating mean, and the shorthands

$$
C = \sum_j w_j \cos\omega t_j,\quad S = \sum_j w_j \sin\omega t_j,\quad
YC = \sum_j w_j y_j \cos\omega t_j,\quad YS = \sum_j w_j y_j \sin\omega t_j,
$$
$$
CC = \sum_j w_j \cos^2\omega t_j - C^2,\quad SS = \sum_j w_j \sin^2\omega t_j - S^2,\quad CS = \sum_j w_j \cos\omega t_j \sin\omega t_j - CS,
$$

the power is

$$
P(f) = \frac{SS\cdot YC^2 + CC\cdot YS^2 - 2\,CS\cdot YC\cdot YS}{YY\,(CC\cdot SS - CS^2)},
\qquad YY = \sum_j w_j (y_j - \bar y)^2 .
$$

Three things to keep in mind:

- The numerator is the squared projection of $y$ on the pair $(\cos\omega t, \sin\omega t)$; the denominator corrects for the pair not being orthogonal under irregular sampling, and normalises by the total variance, so $P \in [0, 1]$.
- Equivalently, $P(f) = 1 - \chi^2(f)/\chi^2_0$: one minus the residual of the best single sinusoid over the residual of a constant.
- The simplest way to see it: with $z_j = w_j\,(y_j - \bar y)$ and ignoring the sampling correction,
$$
P(f) \;\propto\; \Big|\sum_j z_j\, e^{\,i\omega t_j}\Big|^2 .
$$
That is the squared magnitude of a non-uniform Fourier sum of the centred, weighted brightness. The corrections make it exact for one sinusoid; the Fourier sum is the heart of it.

Every quantity is a sum over single points. Nothing is learned. The "feature" of a point is its brightness.

## 2. The spectral layer

Replace the scalar brightness by a learned vector. Each token $j$ carries a feature $h_j \in \mathbb{R}^D$ from the encoder's earlier layers (brightness, error, band, and whatever context the attention has mixed in). A linear map $W \in \mathbb{R}^{D\times M}$ turns it into $a_j = W h_j \in \mathbb{R}^M$, $M$ small (8 to 16). For every trial frequency $f_k$ on a fixed fine grid, $k = 1..K$,

$$
Z_{k,m} = \sum_{j=1}^{N} a_{j,m}\; e^{\,i\,2\pi f_k t_j}, \qquad
S_{k,m} = |Z_{k,m}|^2 \;\;(\text{or } \operatorname{Re}, \operatorname{Im}, |Z| \text{ as separate channels}).
$$

Output: a map $S \in \mathbb{R}^{K\times M}$, one row per frequency, $M$ channels. A small network along the frequency axis (the same convolution we used in the period head) reads the map, and its summary is added to the CLS token. The whole thing is differentiable in $W$ and in $h_j$, so the reconstruction loss trains the encoder *through* the spectrum.

Cost: $N \times K \times M$ multiply-adds per window, with $N \le 256$, $K \approx 10^4$, $M = 8$: about $2\times10^7$, a few milliseconds for a batch on the GPU. Memory: the $K \times M$ map per window, not $N \times K$ (the sum is taken as it goes).

## 3. Where they are the same

Set $M = 1$ and $W$ such that $a_j = w_j (y_j - \bar y)$: the spectral layer's $S_k$ is the squared Fourier sum above, which is Lomb-Scargle up to the sampling correction and the normalisation. Add the correction terms (they are the same kind of sums, with $a_j = w_j$ instead of $w_j y_j$) and the normalisation, and the layer **is** the generalised Lomb-Scargle periodogram. So Lomb-Scargle is one point in the layer's parameter space, and the layer can start there.

Both share:

- the same frequency grid, and the same resolution: the peak width is $1/T$ for a window of length $T$, so a fine grid is needed, about $K = \log(f_{\max}/f_{\min}) / \log(1 + \delta)$ with $\delta$ the relative step;
- the same aliases: a sum over the same times has the same window function, so a 1-day cadence puts the same alias peaks at $f \pm 1$ cycle per day, and a signal at $f$ also shows at its harmonics for non-sinusoidal shapes;
- the same cost shape, points times frequencies.

## 4. Where they differ

| | Lomb-Scargle | spectral layer |
|---|---|---|
| what is summed | the brightness, weighted by $1/\sigma^2$ | a learned vector per point, computed from brightness, error, band and context |
| channels | one | $M$; channels can carry different bands, "dip" vs "bump" shape, or sign conventions, and the reader can combine $S(f)$ with $S(2f)$ to separate a period from its half |
| model of the signal | one sinusoid at $f$ | nothing assumed; the reader network decides what a peak means |
| output | one number per frequency, in $[0,1]$ | $M$ numbers per frequency, unnormalised; the reader learns the scale |
| normalisation | exact ($\chi^2$ ratio) | none unless added; a LayerNorm over frequencies does the job |
| sampling correction | the $CC, SS, CS$ terms | absent by default; can be added as extra channels ($a_j = w_j$), which lets the reader learn the correction |
| what is learned | nothing | $W$, the reader along frequency, and the encoder layers before it |
| gradient | none | flows to every token feature: the encoder is trained to make its features "fold well" |
| harmonics and aliases | handled afterwards by rules | handled by the reader, from data |

The honest summary: the layer keeps Lomb-Scargle's **mechanism** (sum of phasors over the actual times, so irregular sampling is handled exactly and the fine period is resolved) and replaces its **content** (what is summed) and its **read-out** (what the spectrum means) by learned ones. It is a learned periodogram, not a new way to find periodicity.

## 5. Why rotary attention cannot do this

Rotary attention also multiplies by phasors, which is why it looked like it should find periods. In one head, with query $q_i$ and key $k_j$ split into pairs $(q_i^{(r)}, k_j^{(r)})$ for rungs $r = 1..R$ at frequencies $f_r$, the score is

$$
s_{ij} = \sum_{r=1}^{R} \operatorname{Re}\!\Big[\, q_i^{(r)} \overline{k_j^{(r)}}\; e^{\,i\,2\pi f_r (t_i - t_j)}\Big].
$$

Three things kill it as a periodogram:

1. **The rungs are summed before anything is read.** The model sees $s_{ij}$, one number per pair, never the per-rung terms. The spectral layer keeps one output per frequency.
2. **It is pairwise, then softmaxed.** The softmax turns scores into weights for mixing values; the "power at $f_r$" is never formed. Summing $e^{i\omega(t_i - t_j)}$ over all pairs would give $|\sum_j e^{i\omega t_j}|^2$, the right kind of quantity, but attention never sums the scores over pairs, it normalises them per row.
3. **Few rungs.** $R$ per head is the head width times the time fraction over two: 28 per head, 378 to 2016 in the whole model, spaced 0.5 to 4 % apart, against a peak width of 0.1 %. The spectral layer uses $K \approx 10^4$ because its cost is linear in $K$, not quadratic in $N$ per rung.

A wider RoPE helps with point 3 only. That is the test running now. Points 1 and 2 remain whatever the width.

## 6. Has this been done before?

Parts of it, in different places. I found no paper that trains a periodogram-shaped layer end to end inside a light-curve encoder.

- **Differentiable Lomb-Scargle as a layer.** LSCD (Fons et al., ICML 2025) implements Lomb-Scargle in PyTorch and uses it to condition a diffusion model for imputing irregular time series. Same operator, fixed content ($a_j = y_j$), no learned features, used as a conditioning signal rather than inside an encoder. The closest prior art for the mechanics.
- **Learned sinusoidal features of time.** Time2Vec (Kazemi et al. 2019) learns frequencies and phases of sinusoids of $t$ and feeds them to the model as an input encoding. That is a learned *positional* feature per point, not a sum over points: it cannot form a periodogram by itself, the network has to combine points afterwards. Our absolute-time ablation is a fixed-frequency version of it, and it did nothing for the period.
- **Spectral mixture kernels** (Wilson & Adams 2013) parameterise a Gaussian-process covariance as a sum of Gaussians in frequency, and fit the frequencies by maximum likelihood. Mathematically this is a learned periodogram with a handful of peaks, and it works on irregular times. It is a kernel, fit per star by optimisation, not an amortised layer.
- **Frequency-domain transformers for regular series.** FEDformer, FITS, Fredformer and the like apply learned filters to the discrete Fourier transform of a regularly sampled sequence. Regular grid, FFT, filter and transform back: the neural-operator flavour. Not applicable to irregular times, and their goal is smoothing, not period estimation.
- **Non-uniform FFT estimators.** M²NuFFT (2024) and nufftcf (2026) compute power spectra and correlation functions of irregular series fast with the NUFFT. Fixed content, no learning; useful as a faster engine for the sum if $K \times N$ ever becomes the bottleneck.
- **In astronomy.** Networks are routinely given the Lomb-Scargle periodogram, or peaks extracted from it, as *input features* (TESS rotation periods with deep learning, several variable-star classifiers), and one 2017 recurrent network estimated periods from raw irregular curves and was slightly worse than Lomb-Scargle. None put the sum inside the network with learned content.

The paper you linked, arXiv 2402.05871, is "Kilonova Light-Curve Interpolation with Neural Networks": an emulator that maps simulation parameters to smooth kilonova light curves. It does not touch irregular sampling or periodicity, so it does not bear on this question.

## 7. Design choices to settle when building

- **Which features feed the sum.** The token features after the first two or three transformer layers (so band and error are already mixed in), through a linear map to $M = 8$ channels, with one channel pinned to the weighted centred brightness so Lomb-Scargle is always present as a baseline channel.
- **Grid.** Log-spaced, $0.02$ to $500$ d, relative step $5\times10^{-4}$: $K \approx 20{,}000$. Fixed, not learned.
- **Output channels.** $|Z|^2$ per channel plus $\operatorname{Re}Z$, $\operatorname{Im}Z$ for the pinned channel, log-scaled and layer-normalised over frequency.
- **Reader.** The dilated 1-D convolution along frequency from the period head, 64 channels, 4 blocks, then mean-pooled and projected to the model width and added to the CLS token before the remaining layers.
- **Loss.** The masked reconstruction loss as before, with an optional auxiliary cross-entropy on the catalogue period's bin from the reader's output, weight to be tuned; the layer must help the reconstruction on its own for the latent to carry what it finds.
- **Checks.** The period probe (hit rate within 10 %, 1 %, 0.1 %) and the phase probe on the cached latents, and the shuffle score, exactly as for every encoder so far.

## Sources

- Scargle, J. D. 1982, ApJ 263, 835. Zechmeister & Kürster 2009, A&A 496, 577 (generalised Lomb-Scargle). VanderPlas 2018, ApJS 236, 16 (the review).
- LSCD: Lomb-Scargle Conditioned Diffusion for Time series Imputation, Fons et al., ICML 2025: https://arxiv.org/abs/2506.17039
- Time2Vec, Kazemi et al. 2019: https://arxiv.org/abs/1907.05321
- Spectral mixture kernels, Wilson & Adams 2013: https://arxiv.org/abs/1302.4245
- M²NuFFT, 2024: https://arxiv.org/abs/2407.01943 ; nufftcf, 2026: https://arxiv.org/abs/2609.03866
- Recovery of TESS stellar rotation periods using deep learning: https://arxiv.org/abs/2104.14566 ; RNN for unevenly sampled variable stars (2017): https://arxiv.org/abs/1711.10609
- Kilonova light-curve interpolation (the linked paper): https://arxiv.org/abs/2402.05871
