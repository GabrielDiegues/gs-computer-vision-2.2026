"""Avalia o modelo no split de teste e salva métricas e gráficos para o relatório.

Cenários (--cenario):
  completo          todo o split
  noite             só as fotos noturnas reais do split
  noite_sintetica   as fotos diurnas do split convertidas para noite (mesma
                    transformação do treino) — mede se o modelo reconhece um
                    poste apagado sem depender do céu claro
"""
import argparse
import json
import random
import shutil

import cv2
import yaml
from ultralytics import YOLO

from common import CLASSES, DATA_YAML, DATASET_DIR, DOCS_IMG, MODELS_DIR, RUNS_DIR
from app.scene import eh_noturna
from merge_remap import day_to_night


def subset_yaml(split: str, cenario: str):
    """Gera um data.yaml cujo split aponta para o subconjunto do cenário."""
    images = sorted((DATASET_DIR / "images" / split).glob("*.jpg"))
    if cenario == "noite":
        chosen = [str(p) for p in images if eh_noturna(cv2.imread(str(p)))]
    else:  # noite_sintetica
        out = DATASET_DIR / f"{split}_noite_sintetica"
        shutil.rmtree(out, ignore_errors=True)
        (out / "images").mkdir(parents=True)
        (out / "labels").mkdir(parents=True)
        rng, chosen = random.Random(0), []
        for p in images:
            img = cv2.imread(str(p))
            if eh_noturna(img):
                continue
            target = out / "images" / p.name
            cv2.imwrite(str(target), day_to_night(img, rng))
            shutil.copy2(DATASET_DIR / "labels" / split / f"{p.stem}.txt", out / "labels" / f"{p.stem}.txt")
            chosen.append(str(target))
    list_file = DATASET_DIR / f"{split}_{cenario}.txt"
    list_file.write_text("\n".join(chosen))
    data = yaml.safe_load(DATA_YAML.read_text())
    data[split] = str(list_file)
    data_file = DATASET_DIR / f"data_{split}_{cenario}.yaml"
    data_file.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    return data_file, len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(MODELS_DIR / "best.pt"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--cenario", choices=["completo", "noite", "noite_sintetica"], default="completo")
    parser.add_argument("--tag", default="", help="prefixo dos arquivos de saída (ex.: ablacao)")
    args = parser.parse_args()

    if args.cenario == "completo":
        data = DATA_YAML
        n_images = len(list((DATASET_DIR / "images" / args.split).glob("*.jpg")))
    else:
        data, n_images = subset_yaml(args.split, args.cenario)
    tag = "_".join(t for t in [args.tag, args.split, args.cenario] if t)

    model = YOLO(args.weights)
    metrics = model.val(data=str(data), split=args.split, project=str(RUNS_DIR),
                        name=f"eval_{tag}", exist_ok=True, plots=True, verbose=False)

    result = {
        "cenario": tag,
        "pesos": args.weights,
        "imagens": n_images,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precisao": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "por_classe": {},
        "velocidade_ms": {k: round(v, 2) for k, v in metrics.speed.items()},
    }
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        result["por_classe"][CLASSES[int(cls_idx)]] = {
            "precisao": round(float(p), 4), "recall": round(float(r), 4),
            "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    metrics_dir = DOCS_IMG.parent / "metricas"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / f"{tag}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    eval_dir = RUNS_DIR / f"eval_{tag}"
    for name in ["confusion_matrix_normalized.png", "BoxPR_curve.png"]:
        if (eval_dir / name).exists():
            shutil.copy2(eval_dir / name, DOCS_IMG / f"{tag}_{name}")


if __name__ == "__main__":
    main()
