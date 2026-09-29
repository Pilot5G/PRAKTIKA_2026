"""Демо: streamlit run demo_app.py"""
import json, time, datetime
import numpy as np, cv2, streamlit as st, pandas as pd
from common import *
import report

st.set_page_config(page_title="Сегментация спутниковых снимков", layout="wide")
M = report.load_metrics(); names = [m["name"] for m in M["models"]]
st.title(f"Геоаналитика: маски объектов ({'здания' if M['target'] == 'buildings' else 'дороги'})")
name = st.sidebar.selectbox("Модель", names, index=names.index(M["best"]))
thr = st.sidebar.slider("Порог вероятности", .1, .9, .5)
src = st.sidebar.radio("Источник", ["Тестовая выборка", "Загрузить файл"])


@st.cache_resource
def get(n): return load_trained(n)


gt = None
if src == "Тестовая выборка":
    files = sorted((ROOT / "data" / "test" / "images").glob("*.png"))
    f = st.sidebar.selectbox("Снимок", files, format_func=lambda p: p.name)
    img = read_rgb(f); gt = imread(str(ROOT / "data" / "test" / "masks" / f.name), 0) > 127; label = f.name
else:
    up = st.sidebar.file_uploader("Снимок", type=["png", "jpg", "jpeg", "tif"])
    if not up: st.info("Загрузите изображение"); st.stop()
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(up.read(), np.uint8), 1), cv2.COLOR_BGR2RGB); label = up.name
img = cv2.resize(img, (SIZE, SIZE))
if gt is not None and gt.shape[0] != SIZE: gt = cv2.resize(gt.astype(np.uint8), (SIZE, SIZE)) > 0

model = get(name); t = time.perf_counter(); prob = predict_prob(model, img); ms = (time.perf_counter() - t) * 1000
mask = prob > thr
n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
objects = int((stats[1:, cv2.CC_STAT_AREA] >= 10).sum())
conf = float(prob[mask].mean()) if mask.any() else float(1 - prob.mean())
overlay = img.copy(); overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
up_ = lambda a: cv2.resize(a, (384, 384), interpolation=cv2.INTER_NEAREST)

c = st.columns(3)
c[0].image(up_(img), caption="Исходный снимок"); c[1].image(up_(overlay), caption=f"Маска ({name})")
c[2].image(up_((gt.astype(np.uint8) * 255) if gt is not None else (prob * 255).astype(np.uint8)),
           caption="Эталонная разметка" if gt is not None else "Карта вероятностей")
rec = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "source": label, "model": name, "coverage_pct": round(float(mask.mean() * 100), 2),
       "objects": objects, "confidence": round(conf, 3), "ms": round(ms, 1)}
k = st.columns(5); k[0].metric("Покрытие, %", rec["coverage_pct"]); k[1].metric("Объектов", objects)
k[2].metric("Уверенность", f"{conf:.2f}"); k[3].metric("Время, мс", rec["ms"])
if gt is not None:
    iou, d = iou_dice(mask, gt); rec["iou"] = round(iou, 3); k[4].metric("IoU / Boundary F1", f"{iou:.2f} / {boundary_f1(mask, gt):.2f}")
if st.button("Записать в историю (JSON)"):
    report.add_history(rec); st.success("Сохранено в results/history.json")
h = report.load_history()
st.subheader("История запусков")
if h: st.dataframe(pd.DataFrame(h).iloc[::-1], use_container_width=True)
b = st.columns(2)
if b[0].button("Сформировать отчёты PDF/Excel"):
    report.make_excel(); report.make_pdf(); st.success("results/report.pdf и results/report.xlsx готовы")
for col, fn, mime in [(b[0], "report.pdf", "application/pdf"), (b[1], "report.xlsx", "application/vnd.ms-excel")]:
    if (report.RES / fn).exists(): col.download_button(f"Скачать {fn}", open(report.RES / fn, "rb"), fn, mime)
st.subheader("Сравнение архитектур")
st.dataframe(pd.DataFrame(M["models"])[list(report.COLS)].rename(columns=report.COLS).round(3), use_container_width=True)
