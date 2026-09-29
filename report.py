"""Отчёты: Excel и PDF (сравнение архитектур + история запусков)."""
import json, datetime
from pathlib import Path
import pandas as pd
import matplotlib
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).parent; RES = ROOT / "results"
HIST = RES / "history.json"
pdfmetrics.registerFont(TTFont("DV", str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")))

COLS = {"name": "Архитектура", "iou": "IoU", "dice": "Dice", "bf1": "Boundary F1", "ms_per_img": "мс/изобр.",
        "params_m": "Параметры, млн", "size_mb": "Размер, МБ", "train_time_s": "Обучение, с", "score": "Балл"}


def load_metrics(): return json.load(open(RES / "metrics.json", encoding="utf-8"))
def load_history(): return json.load(open(HIST, encoding="utf-8")) if HIST.exists() else []


def add_history(rec):
    h = load_history(); h.append(rec)
    json.dump(h, open(HIST, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def make_excel(path=RES / "report.xlsx"):
    m = load_metrics(); df = pd.DataFrame(m["models"])[list(COLS)].rename(columns=COLS).round(4)
    with pd.ExcelWriter(path) as w:
        df.to_excel(w, sheet_name="Сравнение", index=False)
        pd.DataFrame(load_history()).to_excel(w, sheet_name="История запусков", index=False)
    return path


def make_pdf(path=RES / "report.pdf"):
    m = load_metrics(); hist = load_history()
    st = getSampleStyleSheet()
    H = ParagraphStyle("H", parent=st["Heading2"], fontName="DV"); N = ParagraphStyle("N", parent=st["Normal"], fontName="DV", fontSize=9, leading=13)
    obj = "здания" if m["target"] == "buildings" else "дороги"
    best = next(x for x in m["models"] if x["name"] == m["best"]); ms = m["models"]
    fastest = min(ms, key=lambda x: x["ms_per_img"]); top_iou = max(ms, key=lambda x: x["iou"])
    S = [Paragraph(f"Сегментация: {obj} на спутниковых снимках", ParagraphStyle("T", parent=st["Title"], fontName="DV", fontSize=16)),
         Paragraph(f"Дата: {datetime.datetime.now():%d.%m.%Y %H:%M}. Энкодер: {m['encoder']}, устройство: {m['device']}, эпох: {m['epochs']}.", N),
         Paragraph(f"Данные ({m['dataset']['source']}): {m['dataset']['total']} тайлов {m['dataset']['size']}x{m['dataset']['size']}, "
                   f"бинарная маска (2 класса), split: " + ", ".join(f"{k}={v['count']}" for k, v in m['dataset']['splits'].items()) + ".", N),
         Spacer(1, 8), Paragraph("Сравнение архитектур (тестовая выборка)", H)]
    rows = [list(COLS.values())] + [[f"{x[k]:.3f}" if k != "name" else x[k] for k in COLS] for x in ms]
    t = Table(rows, repeatRows=1); t.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)])); S += [t, Spacer(1, 6)]
    if (RES / "comparison.png").exists(): S.append(Image(str(RES / "comparison.png"), 17 * cm, 4.6 * cm))
    S += [Paragraph("Выбор лучшей модели", H),
          Paragraph(f"Формула балла: {m['score_formula']}. Лучшая: <b>{best['name']}</b> (IoU={best['iou']:.3f}, Boundary F1={best['bf1']:.3f}, "
                    f"{best['ms_per_img']:.1f} мс, {best['size_mb']:.1f} МБ). Максимальный IoU: {top_iou['name']}; самая быстрая: {fastest['name']}. "
                    f"Выбор обусловлен балансом качества региона, точности границ и скорости.", N)]
    ex = RES / f"examples_{m['best']}.png"
    if ex.exists(): S += [Paragraph("Примеры удачной и ошибочной работы (лучшая модель)", H), Image(str(ex), 9 * cm, 13.8 * cm)]
    S.append(Paragraph("Основные ошибки (худшие тайлы по IoU)", H))
    for x in ms:
        S.append(Paragraph(f"{x['name']}: " + "; ".join(f"{w['file']} IoU={w['iou']:.2f} (плотность объектов {w['gt_cov']:.0%})" for w in x["worst"]), N))
    S += [PageBreak(), Paragraph("История запусков демонстрационного модуля", H)]
    if hist:
        hc = ["time", "source", "model", "coverage_pct", "objects", "confidence", "iou"]
        hr = [["Время", "Источник", "Модель", "Покрытие,%", "Объектов", "Уверенность", "IoU"]] + [[str(h.get(c, "-"))[:19] for c in hc] for h in hist[-30:]]
        th = Table(hr, repeatRows=1); th.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), 7), ("GRID", (0, 0), (-1, -1), .4, colors.grey)])); S.append(th)
    else:
        S.append(Paragraph("Запусков пока нет.", N))
    SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm).build(S)
    return path


if __name__ == "__main__":
    print(make_excel(), make_pdf())
