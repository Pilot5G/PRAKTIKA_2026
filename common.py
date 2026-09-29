"""Общие функции: модели, предобработка, метрики."""
from pathlib import Path
import numpy as np, cv2, torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).parent
SIZE = 128                      # размер тайла (маленький -> быстро)
ENCODER = "resnet18"            # лёгкий энкодер
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 5 архитектур сегментации
ARCHS = {
    "Unet": smp.Unet,
    "FPN": smp.FPN,
    "Linknet": smp.Linknet,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "PSPNet": smp.PSPNet,
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_model(name, weights=None):
    return ARCHS[name](encoder_name=ENCODER, encoder_weights=weights, in_channels=3, classes=1)


def load_trained(name):
    m = build_model(name)
    m.load_state_dict(torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu"))
    return m.to(DEVICE).eval()


def imread(path, flag=cv2.IMREAD_COLOR):
    """Чтение файла, работает и с кириллицей в пути (cv2.imread на Windows так не умеет)."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), flag)


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if not ok:
        raise IOError(f"Не удалось закодировать {path}")
    buf.tofile(str(path))


def read_rgb(path):
    return cv2.cvtColor(imread(path), cv2.COLOR_BGR2RGB)


def to_tensor(img_rgb):
    x = (img_rgb.astype(np.float32) / 255 - MEAN) / STD  # нормализация
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


@torch.no_grad()
def predict_prob(model, img_rgb):
    x = to_tensor(img_rgb)[None].to(DEVICE)
    return torch.sigmoid(model(x))[0, 0].cpu().numpy()


# ---------- метрики ----------
def iou_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return (1.0 if union == 0 else inter / union), (1.0 if s == 0 else 2 * inter / s)


def _edge(m):
    m = m.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_f1(pred, gt, tol=2):
    """F1 по границам объектов с допуском tol пикселей (качество границ)."""
    bp, bg = _edge(pred), _edge(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    prec = (bp & cv2.dilate(bg.astype(np.uint8), k).astype(bool)).sum() / bp.sum()
    rec = (bg & cv2.dilate(bp.astype(np.uint8), k).astype(bool)).sum() / bg.sum()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
