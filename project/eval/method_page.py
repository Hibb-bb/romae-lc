"""Build the method page (encoder, decoder, forward pass, training, testing)
as one HTML file with the key figures embedded.

    python -m project.eval.method_page --out project/results/method_page/index.html
"""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path

FIGS = {
    "spectral": "project/results/period_page/figs/spectral_layer.png",
    "forecast": "project/results/forecast_demo_cycrope2/forecast_demo.png",
    "sine": "project/results/sine_demo/sine_demo.png",
    "hits": "project/results/hit_curves/hit_curves.png",
}

HEAD = '''<title>Light-Curve Encoder and Decoder</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* one reading column; figures and tables break out to a wider paper panel; mono for code and numbers */
:root { --bg: #f5f3ee; --fg: #1b1f26; --muted: #5a6270; --rule: #d9d4ca; --accent: #1f6f8b; --panel: #fffdf9; --code: #ece8df;
  --display: "Fraunces", Georgia, serif; --body: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; --mono: "IBM Plex Mono", Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #141821; --fg: #e9ecf1; --muted: #9aa5b5; --rule: #2d3542; --accent: #6fc0da; --panel: #1c2230; --code: #232b3a; color-scheme: dark } }
:root[data-theme="dark"] { --bg: #141821; --fg: #e9ecf1; --muted: #9aa5b5; --rule: #2d3542; --accent: #6fc0da; --panel: #1c2230; --code: #232b3a; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font-family: var(--body); font-size: 16px; line-height: 1.6; margin: 0;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 36px 72px; }
main { max-width: 1080px; margin-inline: auto; display: flex; flex-direction: column; gap: 44px; }
.text { max-width: 68ch; display: flex; flex-direction: column; gap: 12px; min-width: 0; }
h1 { font-family: var(--display); font-weight: 700; font-size: clamp(1.8rem, 4vw, 2.5rem); line-height: 1.12; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 700; font-size: 1.45rem; line-height: 1.25; margin: 0; text-wrap: balance; }
h3 { font-family: var(--body); font-weight: 600; font-size: 1.02rem; margin: 10px 0 0; color: var(--accent); letter-spacing: 0.01em; }
p { margin: 0; }
.lede { color: var(--muted); font-size: 1.08rem; }
section { display: flex; flex-direction: column; gap: 16px; }
figure { margin: 0; background: #fcfcfb; border: 1px solid var(--rule); border-radius: 6px; padding: 8px; overflow-x: auto; }
figure img { display: block; width: 100%; min-width: 0; height: auto; }
figcaption { color: var(--muted); font-size: 0.9rem; padding: 8px 4px 2px; }
ul, ol { margin: 0; padding-left: 1.2em; display: flex; flex-direction: column; gap: 6px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 0.93rem; min-width: 520px; }
th, td { text-align: left; padding: 7px 18px 7px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-family: var(--body); font-weight: 600; color: var(--muted); font-size: 0.78rem; letter-spacing: 0.05em; text-transform: uppercase; }
td.n { text-align: right; padding-right: 24px; }
code { font-family: var(--mono); font-size: 0.88em; background: var(--code); border-radius: 4px; padding: 1px 5px; }
pre { font-family: var(--mono); font-size: 0.86rem; background: var(--code); border-radius: 6px; padding: 12px 14px; overflow-x: auto; margin: 0; line-height: 1.5; }
.note { border-left: 3px solid var(--accent); padding-left: 14px; color: var(--muted); }
.steps { counter-reset: s; display: flex; flex-direction: column; gap: 10px; padding: 0; margin: 0; list-style: none; }
.steps li { display: grid; grid-template-columns: 2.2em 1fr; gap: 10px; align-items: start; }
.steps li::before { counter-increment: s; content: counter(s); font-family: var(--mono); color: var(--accent); font-weight: 500; border: 1px solid var(--rule); border-radius: 50%; width: 1.7em; height: 1.7em; display: grid; place-items: center; font-size: 0.85rem; margin-top: 2px; }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.card { background: var(--panel); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 16px; display: flex; flex-direction: column; gap: 8px; min-width: 0; }
.card h3 { margin: 0; }
a { color: var(--accent); }
</style>
'''


def img(path, alt, caption=None):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b}">{cap}</figure>'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="project/results/method_page/index.html")
    a = ap.parse_args(argv)
    page = f"""{HEAD}<main>
  <div class="text">
    <h1>Light-curve encoder and decoder</h1>
    <p class="lede">How the model reads a ZTF light curve, where the period comes from, how the decoder places a hidden point in the star's cycle, and how all of it is trained and tested. Numbers are from the runs of 1 to 19 October 2026. The results page with the hit-rate tables is <a href="https://claude.ai/artifact/JoubE7CVCj2KSB1QtGzwub">here</a>.</p>
  </div>

  <section>
    <div class="text">
      <h2>The pipeline in one picture</h2>
      <p>A window of observations becomes tokens. The encoder turns them into per-token features and one pooled vector, and on the way its spectral layer finds the star's frequency. The decoder takes the visible tokens' features and answers for the hidden tokens and for any query time, using the phase at the encoder's own frequency. Nothing from a catalogue enters at inference.</p>
    </div>
    <pre class="mermaid">
flowchart LR
  A["250-day window<br/>up to 256 observations"] --> B["tokens<br/>brightness, log error, band, time"]
  B --> C["encoder blocks 1-2<br/>rotary attention"]
  C --> D["spectral layer<br/>20,260 trial frequencies"]
  D --> E["encoder blocks 3-6"]
  D -. "frequency estimate" .-> G
  E --> F["pooled vector (CLS)<br/>period read-out, classes"]
  E --> G["decoder, 2 blocks<br/>phase rope at the estimate"]
  G --> H["brightness at hidden<br/>and query times"]
    </pre>
  </section>

  <section>
    <div class="text">
      <h2>Data and tokens</h2>
      <p>Every star's light curve is standardised per band: subtract the median and divide by 1.4826 times the median absolute deviation, over the whole curve. Times are days since the star's first observation. Training draws sequences of 8 consecutive 250-day windows per star, each window holding between 16 and 256 observations (more are subsampled at random), the gap between windows drawn from 1 to 1.5 window lengths.</p>
      <p>A token is one observation. Its value channels are the standardised brightness and the standardised log of the quoted error. Its position has two axes: time, as days over a fixed unit (the rotary ladder's own scale) plus one, and the band's central wavelength. Nothing else is given: no phase, no period, no class.</p>
      <div class="tablewrap"><table>
        <tr><th>item</th><th>value</th></tr>
        <tr><td>window</td><td>250 days, 16 to 256 tokens, 8 windows per sequence</td></tr>
        <tr><td>value channels</td><td>brightness; standardised log error</td></tr>
        <tr><td>position axes</td><td>time (days / 0.003546 + 1); band wavelength</td></tr>
        <tr><td>bands</td><td>ZTF g, r, i</td></tr>
        <tr><td>training set</td><td>42,622 ZTFxPC stars, 16 fine classes; 5,346 validation stars</td></tr>
      </table></div>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Encoder</h2>
      <p>A 6-block transformer, width 384, 6 heads, 10.7 M parameters, bidirectional over the window's tokens plus one CLS token. Two things make it more than a transformer: the rotary ladder and the spectral layer.</p>
      <h3>Rotary ladder</h3>
      <p>Attention uses continuous rotary position encoding on the time axis. Each head's channels are split into pairs; every pair rotates at its own timescale, and the dot product between two tokens then depends on their time difference at each of those timescales. The ladder is 756 distinct timescales from 0.022 to 500 days, placed at the quantiles of the periods seen in the data, 21 per head, different across the 6 heads and 6 blocks. A small block of channels rotates on the wavelength axis instead, so attention also sees which band a token is in. Three quarters of the channel pairs rotate; the rest carry no position.</p>
      <h3>Spectral layer</h3>
      <p>Placed after block 2. For every token it maps the feature plus the raw value channels to 8 numbers, centres them over the window, and for each of 20,260 trial frequencies (log-spaced from 500 days to half an hour, step 0.05 %) sums them times the phasor at the token's own time and keeps the squared magnitude. That is a learned periodogram: with one channel equal to the brightness it is the Lomb-Scargle periodogram, and that is how channel 0 is initialised. The 8-channel power map, read at f, 2f and f/2 with logs, is standardised and read by a 3-block dilated 1-D convolution along the frequency axis (width 32, kernel 9) that gives one score per bin. The softmax over the bins gives a peak position, a spread and a pooled summary, which are added to the CLS token before blocks 3 to 6.</p>
    </div>
    {img(FIGS["spectral"], "The spectral layer next to Lomb-Scargle", "The layer keeps the phasor sum over the actual times that Lomb-Scargle uses, and learns what is summed and how the spectrum is read.")}
    <div class="text">
      <h3>Outputs</h3>
      <p>Per-token features for the decoder, and the CLS vector pooled over a star's windows (the mean) for the period read-out and the classifier. The spectral layer's score vector is kept as well: it is where the decoder gets its frequency.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Decoder</h2>
      <p>Small by design: 2 transformer blocks of width 192, 3 heads, 0.96 M parameters. Its inputs are the encoder's features of the visible tokens, mapped 384 to 192, plus one learned constant token for every hidden position and every query. A norm and one linear layer give the brightness per hidden token. The decoder's attention uses the encoder's rotary ladder rescaled to its head size.</p>
      <h3>Phase rope</h3>
      <p>On its own this decoder is a copier: it reproduces a hidden point from seen points within about half a cycle, and knows nothing beyond. Placing a point many cycles away needs the time gap divided by the period, keeping the fraction, which attention over fixed timescales cannot do. The phase rope gives every decoder token one more position axis: its time multiplied by the encoder's frequency estimate, its phase in cycles. Eight channels per head are taken from the time ladder and rotate on that axis at 2π times the phase and its 2nd, 3rd and 4th harmonics. Two tokens at the same phase of the star's cycle then line up in the attention score directly, whatever their distance in days.</p>
      <h3>The frequency estimate</h3>
      <p>Read off the spectral layer's scores for the window's visible tokens, detached, so the loss never pushes the estimate through the rope. The expectation of the softmax over all bins is blurred whenever the true peak, its aliases and its harmonics are lit together; the estimator used now is the softmax centroid within 5 bins of the argmax, which is sub-bin and ignores the other peaks.</p>
      <div class="tablewrap"><table>
        <tr><th>estimator, half-masked real windows</th><th>within 0.05 %</th><th>within 5 %</th><th>median error</th></tr>
        <tr><td>softmax expectation over all bins</td><td class=n>4 %</td><td class=n>52 %</td><td class=n>4.3 %</td></tr>
        <tr><td>argmax bin</td><td class=n>65 %</td><td class=n>70 %</td><td class=n>0.025 %</td></tr>
        <tr><td>centroid within 5 bins of the argmax</td><td class=n>66 %</td><td class=n>70 %</td><td class=n>0.011 %</td></tr>
      </table></div>
      <p class="note">The 30 % of windows left are those where the highest peak is an alias or a harmonic of the true frequency. That is the open problem on the decoder side, and it is an estimator problem, not a decoder one.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Forward pass, step by step</h2>
      <ol class="steps">
        <li>Cut the window, standardise, build the tokens and their positions.</li>
        <li>Choose the hidden tokens: half of the real tokens, at random, or as one stretch plus random points. Append the query tokens (training only): extra hidden tokens at times with no observation.</li>
        <li>Encoder on the visible tokens and the CLS: blocks 1 and 2, the spectral layer (its summary is added to the CLS, its scores are kept), blocks 3 to 6.</li>
        <li>Frequency estimate from the kept scores: local centroid around the argmax, detached.</li>
        <li>Decoder input: the visible tokens' features through the 384-to-192 map, one mask token per hidden or query position, every token carrying its time, its band and its phase at the estimate.</li>
        <li>Two decoder blocks with the rotary ladder and the phase rope; the head gives one brightness per hidden or query token.</li>
        <li>Loss on the hidden and query tokens against the target (observations or templates, see below); plus the auxiliary loss on the spectral scores.</li>
      </ol>
      <p>At inference the same pass runs with no queries required: the pooled CLS feeds the period read-out and the classifier, and any set of query times can be appended to get the decoder's brightness at those times. The period and phase are never inputs.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Training</h2>
      <h3>Objective</h3>
      <p>Masked reconstruction: mean squared error on the hidden tokens. Two target choices. Observed: the hidden observations themselves, the plain autoencoder. Template: one Fourier fit per star and band on the whole light curve at the period table's period, with the harmonic count chosen per star from 1, 2, 3, 4, 6 or 8 by adjusted R2; the target at any time, including query times, is that smooth wave. The template is the de-noised brightness and gives the far queries a clean answer to learn.</p>
      <h3>Queries in cycles</h3>
      <p>128 query tokens per window at times with no observation. Each sits at a distance from a random seen token drawn log-uniform in cycles of the star's period, between 0.1 cycle and a cap that ramps geometrically from 5 to 300 cycles over the first 60 % of training; a fifth of the queries are at uniform random times so their layout does not reveal the period. Day-based distances were tried first and fail: 30 days is 100 cycles for an RR Lyrae and a tenth of a cycle for a long-period star.</p>
      <h3>Auxiliary loss on the period</h3>
      <p>A cross-entropy on the spectral layer's scores against the bin of the catalogue period, weight 0.5, with a soft target one bin wide. It is what gives the frequency estimate its precision: the same encoder trained without it reaches 0.72 within 10 % but only 0.19 within 1 %. The catalogue enters here and in the template periods, during training only.</p>
      <h3>Schedule and numerics</h3>
      <div class="tablewrap"><table>
        <tr><th>item</th><th>value</th></tr>
        <tr><td>optimiser</td><td>AdamW, learning rate 1e-4, weight decay 0.05 (none on the phase features), cosine with 2 % warm-up, gradient clip 1.0</td></tr>
        <tr><td>batch</td><td>64 sequences of 8 windows, 512 windows per step (128 sequences for the base encoder)</td></tr>
        <tr><td>steps</td><td>50,000 (100,000 for the base encoder); about 0.5 s per step on one GH200</td></tr>
        <tr><td>precision</td><td>bf16 autocast, except the spectral layer's sum and the decoder's phase angles in fp32</td></tr>
        <tr><td>warm start</td><td>the decoder runs start from the base encoder's weights (every tensor whose name and shape match), with a fresh optimiser</td></tr>
        <tr><td>masking</td><td>random 50 % for the base encoder and the cycle runs; block-plus (a stretch plus random points) for the comparisons</td></tr>
      </table></div>
      <h3>The runs behind the numbers</h3>
      <div class="tablewrap"><table>
        <tr><th>run</th><th>what</th><th>period within 10 % / 1 %</th><th>reach</th></tr>
        <tr><td>maew_spec</td><td>base encoder, observed targets, 100k steps</td><td class=n>0.790 / 0.724</td><td>decoder copies within half a cycle</td></tr>
        <tr><td>cyc</td><td>warm start, templates, queries in cycles</td><td class=n>0.808 / 0.733</td><td>about 20 cycles</td></tr>
        <tr><td>cycrope</td><td>cyc + phase rope, soft estimate</td><td class=n>0.770 / 0.664</td><td>far for the 3 % of windows with a precise estimate</td></tr>
        <tr><td>cycrope2</td><td>cyc + phase rope, local-centroid estimate</td><td class=n>0.751 / 0.230 (pooled read-out)</td><td>far for the 66 % with a precise estimate; see the figure below</td></tr>
      </table></div>
      <p class="note">With the rope the decoder reads the fine frequency straight from the spectral scores, so the pooled vector stops carrying it and the read-out on the pooled vector loses precision. The period should then be read from the spectral peak itself; that probe is being added.</p>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Testing</h2>
      <h3>Period</h3>
      <ol class="steps">
        <li>Cache the pooled vector of every star (mean over its windows at stride a quarter window).</li>
        <li>Read-out: a small MLP on the pooled vector, fitted on the training stars, that classifies the log period into 240 bins and regresses an offset within the bin. This is the model's own period, with its top-5 bins as candidates.</li>
        <li>Fine search ("model → LS"): every period within 10 % of the model's period, 5 trials per one-over-baseline, scored by the fold on the full light curve, then 10 sub-steps around the best. The fold score is the adjusted R2 of a weighted 6-harmonic Fourier fit to the phase-folded curve, one fit per band, pooled: the multi-term Lomb-Scargle statistic.</li>
        <li>Candidates ("(model + aliases) → LS"): the same search seeded by the other top bins and by the double and the half, harmonics scaled so each candidate has the same frequency reach; a candidate replaces the first answer only if its fold is better by 0.02.</li>
        <li>Score: hit rate within 20, 10, 1, 0.1 and 0.01 % of the catalogue period, strict and with the double and the half accepted; the fold R2 as the label-free check.</li>
      </ol>
    </div>
    {img(FIGS["hits"], "Hit rate against the tolerance, strict and alias-tolerant", "All 5,346 validation stars. Lomb-Scargle is the astropy multiband search on a uniform grid; its strict rate stays at a fifth because its best peak is the double or the half for most eclipsing binaries.")}
    <div class="text">
      <h3>Forecasting</h3>
      <p>The decoder is scored by the share of the template's variance it explains at hidden and query points, binned by the distance in cycles from the nearest seen point, and split by the precision of the window's frequency estimate. The sine test runs the same encoder, masking and decoder at a small size on clean sinusoids and on the real stars' templates at random times, to isolate the mechanism from the data.</p>
    </div>
    {img(FIGS["forecast"], "cycrope2 against cyc on six validation stars", "Six validation stars, the last 40 % of a window hidden, dense queries over the first cycles. Grey: the template. Green: cycrope2. Orange: cyc. Top three rows: the spectral estimate is within 0.02 % and cycrope2 tracks the template across the whole hidden stretch, also in the folded view on the right. Bottom three: the estimate landed on an alias about 30 % off, and the decoder drifts out of phase.")}
    {img(FIGS["sine"], "The sine test models", "Clean sinusoids, periods from 0.1 to 25 days, the last half hidden; dense queries near the hidden start and 100 cycles into it. Grey: the plain decoder, a flat line. Purple: the oracle frequency. Green: the phasor features at the model's own frequency, on noisy data.")}
    <div class="text">
      <h3>What the tests established</h3>
      <ul>
        <li>Without an explicit phase coordinate every decoder is a copier, on sines and on real stars; the cycle curriculum alone stretches that to about 20 cycles.</li>
        <li>With the coordinate at the true frequency the decoder is perfect 300 cycles out; at the model's own frequency it is as good as the estimate is precise: 0.97 explained to 20 cycles and 0.88 at 20 to 100 for windows within 0.05 %.</li>
        <li>The pooled vector holds no phase in any encoder; the decoder reads it from the tokens.</li>
        <li>The auxiliary loss on the period bin is what makes the estimate sharp; without it the far reach is gone.</li>
      </ul>
    </div>
  </section>

  <section>
    <div class="text">
      <h2>Code map</h2>
      <div class="tablewrap"><table>
        <tr><th>file</th><th>what</th></tr>
        <tr><td><code>romae_lc/tokenize.py</code>, <code>romae_lc/frames.py</code>, <code>romae_lc/data.py</code></td><td>tokens, windows and sequences, standardisation</td></tr>
        <tr><td><code>romae_lc/model.py</code></td><td>encoder, decoder, the masked forward pass, the phase rope and the frequency estimators</td></tr>
        <tr><td><code>romae_lc/spectral.py</code>, <code>romae_lc/rope.py</code></td><td>the spectral layer; the rotary ladder</td></tr>
        <tr><td><code>project/pretrain_mae.py</code>, <code>project/templates.py</code></td><td>training, targets, queries in cycles, the auxiliary loss, warm start</td></tr>
        <tr><td><code>project/fold.py</code>, <code>project/period_probe.py</code>, <code>project/eval/period_table.py</code></td><td>the fold score, the fine search, the read-out, the period table</td></tr>
        <tr><td><code>project/sine_test.py</code>, <code>project/eval/forecast_demo.py</code></td><td>the sine test; the forecast figures</td></tr>
      </table></div>
    </div>
  </section>
</main>
"""
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"wrote {out}: {len(page) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
