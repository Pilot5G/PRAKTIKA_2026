"""Подготовка данных: (синтетический | реальный) датасет -> train/val/test, проверка разметки.
Реальные данные: положите пары data/raw/images/*.png и data/raw/masks/*.png (одинаковые имена).
"""
import argparse, json, random, shutil
from pathlib import Path
import numpy as np, cv2
from common import SIZE, ROOT, imread, imwrite

DATA = ROOT / "data"


def synth(rng, target):
    """Псевдо-спутниковый тайл: фон, дороги, здания с тенями. Возвращает (img, mask)."""
    S = SIZE
    base = np.array([rng.uniform(80, 110), rng.uniform(100, 130), rng.uniform(60, 85)])
    img = np.ones((S, S, 3), np.float32) * base + rng.normal(0, 8, (S, S, 3))
    img = cv2.GaussianBlur(img, (0, 0), 2)
    roads = np.zeros((S, S), np.uint8)
    for _ in range(rng.integers(1, 4)):
        p1, p2 = rng.integers(0, S, 2), rng.integers(0, S, 2)
        cv2.line(roads, tuple(int(v) for v in (p1[0], 0)), tuple(int(v) for v in (p2[0], S - 1)), 1, int(rng.integers(4, 8)))
        cv2.line(roads, (0, int(p1[1])), (S - 1, int(p2[1])), 1, int(rng.integers(4, 8)))
    img[roads > 0] = rng.uniform(120, 160)
    blds = np.zeros((S, S), np.uint8)
    keep_out = cv2.dilate(roads, np.ones((5, 5), np.uint8))
    for _ in range(int(rng.integers(5, 45))):        # плотность застройки варьируется
        c = rng.integers(8, S - 8, 2)
        w, h, a = rng.integers(7, 20), rng.integers(7, 20), rng.uniform(0, 90)
        box = cv2.boxPoints(((float(c[0]), float(c[1])), (float(w), float(h)), float(a))).astype(np.int32)
        m = np.zeros((S, S), np.uint8); cv2.fillPoly(m, [box], 1)
        if (m & keep_out).any() or (m & cv2.dilate(blds, np.ones((3, 3), np.uint8))).any():
            continue
        cv2.fillPoly(img, [box + 3], (35, 40, 35))    # тень
        col = [(170, 60, 50), (150, 150, 150), (200, 180, 150), (120, 80, 70)][rng.integers(0, 4)]
        cv2.fillPoly(img, [box], col)
        blds |= m
    img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, (blds if target == "buildings" else roads) * 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="сгенерировать учебный датасет")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--target", choices=["buildings", "roads"], default="buildings")
    a = ap.parse_args()
    rng, raw = np.random.default_rng(42), DATA / "raw"
    pairs = []
    if a.synthetic or not (raw / "images").exists():
        print("Генерация синтетического датасета...")
        for i in range(a.n):
            img, m = synth(rng, a.target)
            pairs.append((f"s{i:04d}.png", img, m))
        source = "synthetic"
    else:
        for p in sorted((raw / "images").glob("*.*")):
            mp = raw / "masks" / p.name
            if not mp.exists():
                print("нет маски для", p.name); continue
            img = cv2.resize(imread(str(p)), (SIZE, SIZE))[:, :, ::-1]
            m = cv2.resize(imread(str(mp), 0), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            pairs.append((p.stem + ".png", np.ascontiguousarray(img), m))
        source = "real"
    # проверка разметки
    good = [(n, i, (m > 127).astype(np.uint8) * 255) for n, i, m in pairs if i.shape[:2] == m.shape[:2]]
    print(f"Пар изображение/маска: {len(good)} (отброшено {len(pairs) - len(good)})")
    if not good:
        raise SystemExit("Нет данных: положите пары в data/raw/images и data/raw/masks или запустите с --synthetic")
    random.Random(42).shuffle(good)
    n = len(good); cut = {"train": (0, int(.7 * n)), "val": (int(.7 * n), int(.85 * n)), "test": (int(.85 * n), n)}
    if DATA.exists():
        for s in cut: shutil.rmtree(DATA / s, ignore_errors=True)
    info = {"source": source, "target": a.target, "classes": 2, "size": SIZE, "total": n, "splits": {}}
    for s, (lo, hi) in cut.items():
        (DATA / s / "images").mkdir(parents=True, exist_ok=True); (DATA / s / "masks").mkdir(parents=True, exist_ok=True)
        cov = []
        for name, img, m in good[lo:hi]:
            imwrite(str(DATA / s / "images" / name), img[:, :, ::-1]); imwrite(str(DATA / s / "masks" / name), m)
            cov.append(float((m > 0).mean()))
        info["splits"][s] = {"count": hi - lo, "mean_object_coverage": round(float(np.mean(cov)), 4)}
    json.dump(info, open(DATA / "dataset_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
