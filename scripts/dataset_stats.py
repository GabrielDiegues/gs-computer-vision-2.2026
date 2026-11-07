"""Contagem de imagens/caixas por split e classe + grade de amostras anotadas."""
import argparse
import random
from collections import Counter

import cv2
import numpy as np

from common import CLASSES, DATASET_DIR, DOCS_IMG, SPLITS
from app.scene import eh_noturna

COLORS = {0: (60, 220, 255), 1: (255, 90, 60)}  # BGR: aceso amarelo, apagado azul


def draw(img_path, label_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        c, x, y, bw, bh = line.split()
        c = int(c)
        x0, y0 = int((float(x) - float(bw) / 2) * w), int((float(y) - float(bh) / 2) * h)
        x1, y1 = int((float(x) + float(bw) / 2) * w), int((float(y) + float(bh) / 2) * h)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[c], max(2, w // 200))
        cv2.putText(img, CLASSES[c], (x0, max(15, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, w / 900), COLORS[c], 2)
    return cv2.resize(img, (320, 320))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=4, help="lado da grade de amostras")
    args = parser.parse_args()
    random.seed(0)

    cols = [f"{c}/{cena}" for c in CLASSES for cena in ("dia", "noite")]
    print("Caixas por classe e cena (dia/noite pelo brilho médio da foto)")
    print(f"{'split':6} {'imagens':>8} {'sintét.':>8} " + " ".join(f"{c:>20}" for c in cols))
    for split in SPLITS:
        labels = sorted((DATASET_DIR / "labels" / split).glob("*.txt"))
        boxes, synthetic = Counter(), 0
        for lp in labels:
            text = lp.read_text().strip()
            synthetic += lp.stem.endswith(("_noite", "_cinza"))
            if not text:
                continue
            img = cv2.imread(str(next((DATASET_DIR / "images" / split).glob(f"{lp.stem}.*"))))
            cena = "noite" if eh_noturna(img) else "dia"
            boxes.update(f"{CLASSES[int(l.split()[0])]}/{cena}" for l in text.splitlines())
        print(f"{split:6} {len(labels):>8} {synthetic:>8} " + " ".join(f"{boxes[c]:>20}" for c in cols))

    train_labels = [lp for lp in (DATASET_DIR / "labels" / "train").glob("*.txt") if lp.read_text().strip()]
    picks = random.sample(train_labels, min(args.grid ** 2, len(train_labels)))
    tiles = []
    for lp in picks:
        img = next((DATASET_DIR / "images" / "train").glob(f"{lp.stem}.*"))
        tiles.append(draw(img, lp))
    while len(tiles) < args.grid ** 2:
        tiles.append(np.zeros((320, 320, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + args.grid]) for i in range(0, len(tiles), args.grid)]
    DOCS_IMG.mkdir(parents=True, exist_ok=True)
    out = DOCS_IMG / "amostras_dataset.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"\nGrade de amostras: {out}")


if __name__ == "__main__":
    main()
