"""Evalúa la regla de violencia con RWF-2000 y calibra su umbral.

RWF-2000 (Cheng et al., 2021) tiene clips de 5 s de cámaras de seguridad reales, en las carpetas
train/ y val/, cada una con Fight/ y NonFight/. Se solicita a sus autores (github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection).

Uso:
  python scripts/evaluar_rwf2000.py --datos D:/RWF-2000/val --max-videos 100
  python scripts/evaluar_rwf2000.py --datos D:/RWF-2000/val --umbrales 1.4 1.6 1.8 2.0 2.4

La pose de cada video se calcula una sola vez y luego se prueban todos los umbrales. Se reportan
accuracy, precisión, recall y F1 (clase positiva = Fight) y los FPS del modelo de pose en esta CPU:
las métricas que la revisión pidió reportar juntas.
"""
from __future__ import annotations

import argparse
import copy
import csv
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from vigia.historial import Historial  # noqa: E402
from vigia.modelos import ruta_modelo  # noqa: E402
from vigia.reglas import Reglas  # noqa: E402
from vigia.tipos import Observacion, Persona  # noqa: E402


def poses_de_video(ruta: Path, modelo_ruta: str, fps: float, imgsz: int, ancho_max: int):
    from ultralytics import YOLO
    modelo = YOLO(modelo_ruta)          # instancia nueva: el seguidor empieza limpio en cada clip
    cap = cv2.VideoCapture(str(ruta))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    paso = max(1, round(fps_video / fps))
    obs, i, t_modelo, n_modelo = [], 0, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % paso == 0:
            if frame.shape[1] > ancho_max:
                e = ancho_max / frame.shape[1]
                frame = cv2.resize(frame, (ancho_max, int(frame.shape[0] * e)))
            t0 = time.perf_counter()
            r = modelo.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=imgsz, conf=0.4,
                             classes=[0], verbose=False)[0]
            t_modelo += time.perf_counter() - t0
            n_modelo += 1
            o = Observacion(t=i / fps_video)
            if r.boxes is not None and r.boxes.id is not None and r.keypoints is not None:
                kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                for j, pid in enumerate(r.boxes.id.int().cpu().tolist()):
                    o.personas.append(Persona(int(pid), r.boxes.xyxy.cpu().numpy()[j],
                                              r.keypoints.xy.cpu().numpy()[j], None if kc is None else kc[j]))
            obs.append(o)
        i += 1
    cap.release()
    return obs, (n_modelo / t_modelo if t_modelo else 0.0)


def predecir(obs, cfg_reglas, umbral):
    r = copy.deepcopy(cfg_reglas)
    for k in r:
        if isinstance(r[k], dict) and "activa" in r[k]:
            r[k]["activa"] = k == "violencia"
    r["violencia"]["velocidad_extremidades"] = umbral
    reglas, hist = Reglas(r), Historial()
    for o in obs:
        hist.actualizar(o)
        if any(e.tipo == "violencia" for e in reglas.evaluar(o, hist, 1, 1, [])):
            return 1
    return 0


def metricas(y, p):
    y, p = np.array(y), np.array(p)
    tp, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 0)).sum())
    fp, fn = int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpeta con Fight/ y NonFight/ (p. ej. RWF-2000/val)")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = todos; si no, muestra balanceada")
    ap.add_argument("--umbrales", type=float, nargs="+", default=[1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8])
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--salida", default=str(RAIZ / "resultados_rwf2000.csv"))
    a = ap.parse_args()

    cfg = yaml.safe_load((RAIZ / "config.yaml").read_text(encoding="utf-8"))
    base = Path(a.datos)
    videos = [(p, 1) for p in sorted((base / "Fight").glob("*.*"))] + \
             [(p, 0) for p in sorted((base / "NonFight").glob("*.*"))]
    if not videos:
        sys.exit(f"No hay videos en {base}/Fight ni {base}/NonFight")
    if a.max_videos:
        random.seed(0)
        pel = [v for v in videos if v[1]]
        nop = [v for v in videos if not v[1]]
        k = a.max_videos // 2
        videos = random.sample(pel, min(k, len(pel))) + random.sample(nop, min(k, len(nop)))

    modelo = ruta_modelo(cfg["modelos"]["pose"])
    filas, etiquetas, preds, fps_l = [], [], {u: [] for u in a.umbrales}, []
    for n, (ruta, y) in enumerate(videos, 1):
        obs, fps_mod = poses_de_video(ruta, modelo, a.fps, cfg["modelos"]["tamano_imagen"],
                                      cfg["procesamiento"]["ancho_max"])
        fps_l.append(fps_mod)
        etiquetas.append(y)
        fila = {"video": ruta.name, "etiqueta": "Fight" if y else "NonFight", "fps_pose": round(fps_mod, 1)}
        for u in a.umbrales:
            p = predecir(obs, cfg["reglas"], u)
            preds[u].append(p)
            fila[f"pred_{u}"] = p
        filas.append(fila)
        print(f"[{n}/{len(videos)}] {ruta.name} ({fila['etiqueta']}) fps={fps_mod:.1f}")

    with open(a.salida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    print(f"\n{len(videos)} videos · {sum(etiquetas)} con pelea · pose a {np.mean(fps_l):.1f} FPS en esta CPU\n")
    print(f"{'umbral':>7} {'accuracy':>9} {'precisión':>10} {'recall':>7} {'F1':>6}   TP  FP  FN  TN")
    mejor = None
    for u in a.umbrales:
        m = metricas(etiquetas, preds[u])
        print(f"{u:7.1f} {m['accuracy']:9.3f} {m['precision']:10.3f} {m['recall']:7.3f} {m['f1']:6.3f}   "
              f"{m['tp']:3d} {m['fp']:3d} {m['fn']:3d} {m['tn']:3d}")
        if mejor is None or m["f1"] > mejor[1]["f1"]:
            mejor = (u, m)
    print(f"\nMejor umbral por F1: {mejor[0]} (F1 = {mejor[1]['f1']:.3f}). "
          f"Póngalo en config.yaml -> reglas.violencia.velocidad_extremidades.")
    print(f"Detalle por video: {a.salida}")


if __name__ == "__main__":
    main()
