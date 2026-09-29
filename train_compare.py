"""Обучение 5 архитектур, оценка (IoU, Dice, Boundary-F1, скорость, размер), примеры ошибок."""
import argparse, json, time, random
import numpy as np, cv2, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from common import *

RES = ROOT / "results"; (ROOT / "models").mkdir(exist_ok=True); RES.mkdir(exist_ok=True)


class DS(Dataset):
    def __init__(self, split, aug=False):
        self.imgs = sorted((ROOT / "data" / split / "images").glob("*.png"))
        self.mdir = ROOT / "data" / split / "masks"; self.aug = aug

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]; img = read_rgb(p); m = (imread(str(self.mdir / p.name), 0) > 127).astype(np.float32)
        if self.aug:  # аугментация: флипы, повороты, яркость
            if random.random() < .5: img, m = img[:, ::-1], m[:, ::-1]
            if random.random() < .5: img, m = img[::-1], m[::-1]
            k = random.randint(0, 3); img, m = np.rot90(img, k), np.rot90(m, k)
            img = np.clip(img.astype(np.float32) * random.uniform(.8, 1.2) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(m))[None]


@torch.no_grad()
def val_iou(model, dl):
    model.eval(); r = []
    for x, y in dl:
        p = (torch.sigmoid(model(x.to(DEVICE))) > .5).cpu().numpy().astype(bool); g = y.numpy().astype(bool)
        r += [iou_dice(p[i, 0], g[i, 0])[0] for i in range(len(p))]
    return float(np.mean(r))


def train(name, weights, epochs, bs):
    model = build_model(name, weights).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    tl = DataLoader(DS("train", True), bs, shuffle=True); vl = DataLoader(DS("val"), bs)
    best, t0 = -1, time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE); opt.zero_grad()
            out = model(x); (dice(out, y) + bce(out, y)).backward(); opt.step()
        v = val_iou(model, vl)
        if v > best: best = v; torch.save(model.state_dict(), ROOT / "models" / f"{name}.pt")
        print(f"  {name} epoch {ep + 1}/{epochs} val_IoU={v:.3f}")
    return best, time.time() - t0


def evaluate(name):
    model = load_trained(name); files = sorted((ROOT / "data" / "test" / "images").glob("*.png"))
    for f in files[:5]: predict_prob(model, read_rgb(f))           # прогрев
    rows, times = [], []
    for f in files:
        img = read_rgb(f); gt = imread(str(ROOT / "data" / "test" / "masks" / f.name), 0) > 127
        if DEVICE == "cuda": torch.cuda.synchronize()
        t = time.perf_counter(); prob = predict_prob(model, img)
        if DEVICE == "cuda": torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
        pred = prob > .5; iou, d = iou_dice(pred, gt)
        rows.append({"file": f.name, "iou": iou, "dice": d, "bf1": boundary_f1(pred, gt), "gt_cov": float(gt.mean())})
    return model, rows, float(np.mean(times))


def save_examples(model, name, rows):
    order = sorted(rows, key=lambda r: r["iou"]); pick = [("Худшие", r) for r in order[:3]] + [("Лучшие", r) for r in order[-3:]]
    fig, ax = plt.subplots(len(pick), 3, figsize=(7, 2.3 * len(pick)))
    for i, (tag, r) in enumerate(pick):
        img = read_rgb(ROOT / "data" / "test" / "images" / r["file"])
        gt = imread(str(ROOT / "data" / "test" / "masks" / r["file"]), 0) > 127
        pr = predict_prob(model, img) > .5
        for j, (im, t) in enumerate([(img, "снимок"), (gt, "разметка"), (pr, f"{name} IoU={r['iou']:.2f}")]):
            ax[i, j].imshow(im, cmap="gray"); ax[i, j].axis("off"); ax[i, j].set_title(f"{tag}: {t}" if j == 0 else t, fontsize=8)
    plt.tight_layout(); plt.savefig(RES / f"examples_{name}.png", dpi=90); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--weights", default=None, help="imagenet (нужен интернет) или пусто")
    a = ap.parse_args(); random.seed(0); torch.manual_seed(0)
    if not list((ROOT / "data" / "train" / "images").glob("*.png")):
        raise SystemExit("Папка data/train пуста. Сначала выполните: python prepare_data.py --synthetic --n 400 --target buildings")
    info = json.load(open(ROOT / "data" / "dataset_info.json", encoding="utf-8")); out = []
    for name in ARCHS:
        print("==", name)
        v, tt = train(name, a.weights, a.epochs, a.bs)
        model, rows, ms = evaluate(name); save_examples(model, name, rows)
        out.append({"name": name, "iou": float(np.mean([r["iou"] for r in rows])), "dice": float(np.mean([r["dice"] for r in rows])),
                    "bf1": float(np.mean([r["bf1"] for r in rows])), "ms_per_img": ms, "val_iou": v, "train_time_s": tt,
                    "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
                    "size_mb": (ROOT / "models" / f"{name}.pt").stat().st_size / 2**20,
                    "worst": [{"file": r["file"], "iou": r["iou"], "gt_cov": r["gt_cov"]} for r in sorted(rows, key=lambda r: r["iou"])[:3]]})
    tmin = min(m["ms_per_img"] for m in out)
    for m in out:  # итоговый балл: качество региона, границ и скорость
        m["score"] = 0.6 * m["iou"] + 0.3 * m["bf1"] + 0.1 * tmin / m["ms_per_img"]
    best = max(out, key=lambda m: m["score"])["name"]
    json.dump({"target": info["target"], "encoder": ENCODER, "device": DEVICE, "epochs": a.epochs, "bs": a.bs, "weights": a.weights, "dataset": info,
               "score_formula": "0.6*IoU + 0.3*BoundaryF1 + 0.1*(t_min/t)", "best": best, "models": out},
              open(RES / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    for k, (key, t) in enumerate([("iou", "IoU"), ("bf1", "Boundary F1"), ("ms_per_img", "мс / изображение")]):
        ax[k].bar([m["name"] for m in out], [m[key] for m in out]); ax[k].set_title(t); ax[k].tick_params(axis="x", rotation=30)
    plt.tight_layout(); plt.savefig(RES / "comparison.png", dpi=90); plt.close()
    print("Лучшая модель:", best)
    import report; report.make_excel(); report.make_pdf()


if __name__ == "__main__":
    main()
