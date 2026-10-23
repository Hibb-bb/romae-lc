#!/bin/bash
# Dennis's synthetic LeJEPA test bed (top-level lcjepa.py, controls.py, diag.py; the
# notebook lightcurve_time_jepa.ipynb cells 4 to 16) as one GPU job. Outputs land in
# /work/hdd/bfrf/hibb/scratch/lcjepa_synth: results.json, controls.json, diag_*.json, ckpt_*.pt.
#SBATCH --job-name=lcjepa-synth
#SBATCH --account=bfrf-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=01:50:00
#SBATCH --chdir=/projects/bfrf/hibb/romae-lc
#SBATCH --output=project/logs/%x-%j.out
#SBATCH --error=project/logs/%x-%j.err
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONPATH=/projects/bfrf/hibb/romae-lc
source .venv/bin/activate
OUT=${OUT:-/work/hdd/bfrf/hibb/scratch/lcjepa_synth}
cp lcjepa.py controls.py diag.py "$OUT"/
cd "$OUT"
python - <<'PY'
import numpy as np, torch, json, time, pandas as pd
from lcjepa import *
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
STEPS, B = 4000, 128
DATA_KW = dict(nightly=False)
VARIANTS = ["lejepa", "lejepa-noerr", "mae", "mjepa-b", "mjepa-2",
            "rot+batch", "rot+time", "rot+batch+int-noerr", "rot+batch+int-noerr+nce"]
test = make_batch(1000, rng=np.random.default_rng(123), **DATA_KW)
w = window_w_ls(to_torch(test, device))[:, 0].cpu().numpy()
sel = test["cls"] != 3
r = w[sel] / (2 * np.pi / 10 ** test["logP"][sel])
for k in range(3):
    rr = r[test["cls"][sel] == k]
    print(f"LS {CLASS_NAMES[k]:25s} peak at f: {np.mean(np.abs(rr-1)<0.02):.2f}   peak at 2f: {np.mean(np.abs(rr-2)<0.04):.2f}", flush=True)
results = {}
for var in VARIANTS:
    t0 = time.time()
    model, hist = train(var, steps=STEPS, B=B, device=device, data_kw=DATA_KW, log_every=500)
    res = evaluate(model, test, device=device)
    res["train_minutes"] = (time.time() - t0) / 60
    results[var] = res
    torch.save(model.state_dict(), f"ckpt_{var}.pt")
    json.dump(results, open("results.json", "w"), indent=1)
    print("####", var, {k: round(v, 3) for k, v in res.items()}, flush=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(pd.DataFrame(results).round(3).to_string(), flush=True)
PY
echo "#### controls"; python controls.py lejepa,lejepa-noerr,mae,mjepa-b,mjepa-2,rot+batch,rot+time,rot+batch+int-noerr,rot+batch+int-noerr+nce
echo "#### diag"; python diag.py lejepa,lejepa-noerr,mae,mjepa-b,rot+batch,rot+batch+int-noerr,rot+batch+int-noerr+nce
echo "exit $?"
