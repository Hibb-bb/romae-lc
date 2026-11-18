import argparse, json, sys
sys.path.insert(0, "/projects/bfrf/hibb/romae-lc-cleanup")
sys.path.insert(0, sys.argv[1])
from recipe import apply_recipe
from project.common import add_data_args, add_frame_args, add_ladder_args, add_model_args, resolve_model_args
from project.pretrain_mae import add_train_args, RUN_CONTROL
from project.tracking import add_wandb_args
def build():
    p = argparse.ArgumentParser()
    for f in (add_data_args, add_frame_args, add_ladder_args, add_model_args, add_train_args, add_wandb_args): f(p)
    return p
P = sys.argv[1]
launcher = "--out project/runs/enc_o3_ph --workers 8 --pin-memory --eval-every 2500 --ckpt-every 500 --time-budget 30000 --wandb".split()
a = resolve_model_args(apply_recipe(build(), ["--recipe", f"{P}/recipes/enc_o3_ph.toml"] + launcher))
ref = json.load(open("/projects/bfrf/hibb/romae-lc/project/runs/enc_o3_ph/args.json"))
d = {k: v for k, v in vars(a).items() if k != "recipe"}
norm = lambda v: list(v) if isinstance(v, tuple) else v
diff = {k: (norm(d.get(k)), ref.get(k)) for k in set(d) | set(ref) if norm(d.get(k)) != ref.get(k)}
print("diff vs args.json:", diff)
b = apply_recipe(build(), ["--recipe", f"{P}/recipes/enc_o3_ph.toml", "--mask-ratio", "0.8", "--forecast-tail", "10", "60"])
print("cli override:", b.mask_ratio, b.forecast_tail, b.steps)
try:
    open(f"{P}/recipes/bad.toml","w").write('extends="enc_o3.toml"\nmask_raito=0.8\n')
    apply_recipe(build(), ["--recipe", f"{P}/recipes/bad.toml"])
except SystemExit as e: print("typo caught, exit", e.code)
