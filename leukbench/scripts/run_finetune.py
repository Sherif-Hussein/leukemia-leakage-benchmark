"""M5: fine-tuned EfficientNet (end-to-end) with stain augmentation. Needs a GPU for real runs.

Same protocol as run_cell_benchmark.py (outer folds by split type, inner validation for early
stopping / threshold / temperature, optional held-out test), so its numbers sit in the same tables.

  python scripts/run_finetune.py --manifest manifests/cnmc.csv --task cnmc_all_vs_hem \
     --splits image grouped --cv-where cohort=train --test-where cohort=prelim_test \
     --arch efficientnet_b1 --epochs 10 --out results/e1_cnmc_m5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package root (works with python -I)
from leukbench import metrics as MT
from leukbench.encoders import IMAGENET_MEAN, IMAGENET_STD
from leukbench.runner import parse_kv, save_json
from leukbench.splits import make_folds
from leukbench.stain import HEDJitter
from leukbench.tasks import load_tasks, select

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--tasks", default=str(Path(__file__).resolve().parents[1] / "configs" / "tasks.json"))
ap.add_argument("--task", required=True)
ap.add_argument("--arch", default="efficientnet_b1")
ap.add_argument("--weights", help="local ImageNet weights file for timm (optional)")
ap.add_argument("--splits", nargs="+", default=["grouped"], choices=["image", "grouped"])
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--only-folds", nargs="*", type=int, help="run a subset of folds (for time limits)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cv-where", nargs="*")
ap.add_argument("--test-where", nargs="*")
ap.add_argument("--no-stain-aug", action="store_true")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
man = pd.read_csv(a.manifest, keep_default_na=False)
spec = load_tasks(a.tasks)[a.task]
cv = select(man, spec, parse_kv(a.cv_where))
test = select(man, spec, parse_kv(a.test_where)) if a.test_where else None

norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
aug = [transforms.RandomResizedCrop(a.size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip(), transforms.RandomRotation(180)]
if not a.no_stain_aug:
    aug = [HEDJitter(0.05, 0.02)] + aug + [transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)]
t_train = transforms.Compose(aug + [transforms.ToTensor(), norm])
t_eval = transforms.Compose([transforms.Resize((a.size, a.size)), transforms.ToTensor(), norm])


class Cells(Dataset):
    def __init__(self, rows, y, t):
        self.paths, self.y, self.t = man.loc[rows, "path"].tolist(), y, t

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.t(Image.open(self.paths[i]).convert("RGB")), float(self.y[i])


def predict(model, rows, y):
    dl = DataLoader(Cells(rows, y, t_eval), batch_size=a.batch_size * 2, num_workers=a.workers)
    model.eval()
    ps = []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for x, _ in dl:
            ps.append(torch.sigmoid(model(x.to(dev)).float().squeeze(1)).cpu().numpy())
    return np.concatenate(ps)


def train(rows, y, groups, kind):
    import timm
    if kind == "grouped":
        tr, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=a.seed).split(rows, y, groups))
    else:
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=a.seed).split(rows, y))
    kw = {"pretrained_cfg_overlay": dict(file=a.weights)} if a.weights else {}
    model = timm.create_model(a.arch, pretrained=True, num_classes=1, **kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=max(1, len(tr) // a.batch_size))
    pos_w = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    dl = DataLoader(Cells(rows[tr], y[tr], t_train), batch_size=a.batch_size, shuffle=True,
                    num_workers=a.workers, drop_last=True)
    best, best_state = np.inf, None
    for ep in range(a.epochs):
        model.train()
        for step, (x, t) in enumerate(dl):
            with torch.autocast(dev, enabled=dev == "cuda"):
                loss = loss_fn(model(x.to(dev)).squeeze(1), t.float().to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step < sched.total_steps - 1:
                sched.step()
            if a.max_steps and step + 1 >= a.max_steps:
                break
        p_va = predict(model, rows[va], y[va])
        vloss = -np.mean(y[va] * np.log(p_va + 1e-7) + (1 - y[va]) * np.log(1 - p_va + 1e-7))
        print(f"    epoch {ep + 1}/{a.epochs} val loss {vloss:.4f}", flush=True)
        if vloss < best:
            best, best_state = vloss, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    p_va = predict(model, rows[va], y[va])
    temp = MT.fit_temperature(y[va], p_va)
    thr = MT.threshold_for_sensitivity(y[va], MT.apply_temperature(p_va, temp), 0.95)
    return model, temp, thr


summary, preds = [], []
for kind in a.splits:
    fold = make_folds(cv.y, cv.groups, kind, a.folds, a.seed)
    oof, thr_f, done = np.full(len(cv.y), np.nan), np.zeros(len(cv.y)), np.zeros(len(cv.y), bool)
    for k in (a.only_folds if a.only_folds is not None else range(a.folds)):
        t0 = time.time()
        tr, te = fold != k, fold == k
        model, temp, thr = train(cv.index[tr], cv.y[tr], cv.groups[tr], kind)
        oof[te] = MT.apply_temperature(predict(model, cv.index[te], cv.y[te]), temp)
        thr_f[te], done[te] = thr, True
        print(f"  {kind} fold {k} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    res = MT.bootstrap(cv.y[done], oof[done], cv.groups[done], thr_f[done], a.n_boot, a.seed)
    summary.append(dict(model="M5", blocks=a.arch, clf="finetune", split=kind, seed=a.seed, eval="cv",
                        n=int(done.sum()), patients=len(np.unique(cv.groups[done])), **res))
    preds.append(pd.DataFrame(dict(model="M5", split=kind, seed=a.seed, eval="cv", row=cv.index[done],
                                   patient_id=cv.groups[done], y=cv.y[done], p=oof[done])))
    print(f"  M5 {kind} CV AUROC {res['auroc']:.3f} [{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]", flush=True)
    if test is not None:
        model, temp, thr = train(cv.index, cv.y, cv.groups, kind)
        p_te = MT.apply_temperature(predict(model, test.index, test.y), temp)
        res = MT.bootstrap(test.y, p_te, test.groups, thr, a.n_boot, a.seed)
        summary.append(dict(model="M5", blocks=a.arch, clf="finetune", split=kind, seed=a.seed, eval="test",
                            n=len(test.y), patients=len(np.unique(test.groups)), **res))
        preds.append(pd.DataFrame(dict(model="M5", split=kind, seed=a.seed, eval="test", row=test.index,
                                       patient_id=test.groups, y=test.y, p=p_te)))
        print(f"  M5 {kind} TEST AUROC {res['auroc']:.3f} [{res['auroc_lo']:.3f}, {res['auroc_hi']:.3f}]", flush=True)

pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
pd.concat(preds).to_csv(out / "predictions.csv.gz", index=False)
save_json(vars(a), out / "config.json")
