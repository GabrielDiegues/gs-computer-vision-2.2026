"""Une os datasets brutos em um único dataset YOLO com as 2 classes do Rua Acesa.

- Remapeia os nomes de classe originais para poste_aceso (0) / poste_apagado (1).
- Descarta imagens com caixas de classe desconhecida (poste sem estado definido),
  para não ensinar o modelo que aquele poste é "fundo".
- Remove duplicatas exatas (hash do arquivo) e quase duplicatas entre datasets
  (hash perceptual dHash), para a mesma foto não aparecer no treino e no teste.
- Agrupa as variações aumentadas do Roboflow (sufixo .rf.<hash>) pela foto de
  origem e mantém só uma por grupo: evita vazamento entre splits e deixa a
  augmentação a cargo do YOLO durante o treino.
- Faz split 70/20/10 estratificado pela presença de poste_apagado.
- Só no treino, corrige dois atalhos que os dados públicos oferecem ao modelo:
  * noite sintética: quase todo poste_apagado público é foto diurna, então parte
    delas recebe uma versão escurecida (o modelo precisa olhar a lâmpada, não o céu);
  * tons de cinza: um dos datasets é todo em cinza e quase só tem poste_apagado,
    então parte das fotos coloridas com poste_aceso recebe uma cópia em cinza.
"""
import argparse
import hashlib
import random
import re
import shutil
from collections import Counter, defaultdict

import cv2
import numpy as np
import yaml

from pathlib import Path

from common import CLASSES, DATASET_DIR, IMG_EXTS, RAW_DIR, SPLITS
from app.scene import eh_noturna

OFF = re.compile(r"off|apagad|broken|fault|dark|unlit|not.?working", re.I)
ON = re.compile(r"(^|[^a-z])on([^a-z]|$)|(^|[^a-z])lit$|aceso|working|glow", re.I)


def map_class(name: str) -> int | None:
    """Converte o nome de classe original para o índice do Rua Acesa (None = descartar)."""
    if OFF.search(name):
        return 1
    if ON.search(name):
        return 0
    return None


def source_key(stem: str) -> str:
    """'foto_123_jpg.rf.abc123' -> 'foto_123_jpg' (agrupa augmentations da mesma foto)."""
    return stem.split(".rf.")[0]


def read_labels(path, class_map):
    """Lê um .txt YOLO e devolve (linhas remapeadas, teve_classe_desconhecida)."""
    lines, unknown = [], False
    if not path.exists():
        return lines, unknown
    for raw in path.read_text().splitlines():
        parts = raw.split()
        if not parts:
            continue
        new_cls = class_map.get(int(parts[0]))
        if new_cls is None:
            unknown = True
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) > 4:  # polígono -> caixa
            xs, ys = coords[0::2], coords[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            coords = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
        lines.append(f"{new_cls} " + " ".join(f"{c:.6f}" for c in coords[:4]))
    return lines, unknown


def dhash(img) -> np.uint64:
    """Hash perceptual de 64 bits: compara o brilho de pixels vizinhos numa miniatura 9x8."""
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten()).view(">u8")[0]


def black_fraction(img) -> float:
    """Fração de pixels pretos puros (bordas de rotação do Roboflow)."""
    return float((img.max(axis=2) < 3).mean())


def day_to_night(img, rng: random.Random):
    """Simula a mesma cena à noite: escurece com curva gama, puxa para o azul e
    adiciona ruído de sensor. Uma lâmpada apagada continua sem brilho próprio."""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(2.0, 3.0) * rng.uniform(0.35, 0.6)
    x *= np.array([rng.uniform(1.05, 1.25), 1.0, rng.uniform(0.75, 0.9)], np.float32)  # BGR
    x = x * 255.0 + np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 6), x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def is_gray(img) -> bool:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()) < 8


def collect(max_background: float, near_dup_bits: int):
    """Lê todos os datasets brutos e devolve uma imagem por foto de origem."""
    groups, seen_hashes, report = defaultdict(list), set(), {}
    for ds_dir in sorted(p for p in RAW_DIR.iterdir() if p.is_dir()):
        meta = yaml.safe_load((ds_dir / "data.yaml").read_text())
        names = meta["names"] if isinstance(meta["names"], list) else list(meta["names"].values())
        class_map = {i: map_class(n) for i, n in enumerate(names)}
        stats = Counter()
        for img in ds_dir.rglob("*"):
            if img.suffix.lower() not in IMG_EXTS or img.parent.name != "images":
                continue
            label = img.parent.parent / "labels" / f"{img.stem}.txt"
            lines, unknown = read_labels(label, class_map)
            if unknown and not lines:
                stats["descartada_classe_desconhecida"] += 1
                continue
            digest = hashlib.md5(img.read_bytes()).hexdigest()
            if digest in seen_hashes:
                stats["duplicada"] += 1
                continue
            seen_hashes.add(digest)
            groups[f"{ds_dir.name}/{source_key(img.stem)}"].append(
                {"img": img, "lines": lines, "prefix": ds_dir.name.split("__")[1]})
            stats["lida"] += 1
        report[ds_dir.name] = {
            "classes": {n: (CLASSES[c] if c is not None else "descartada")
                        for n, c in zip(names, class_map.values())},
            **stats,
        }

    items, hashes = [], np.empty(0, np.uint64)
    for key, members in groups.items():
        # Entre as variações da mesma foto, fica a com menos borda preta (menos rotacionada).
        best = min(members, key=lambda m: black_fraction(cv2.imread(str(m["img"]))))
        ds = key.split("/")[0]
        h = dhash(cv2.imread(str(best["img"])))
        if hashes.size and np.bitwise_count(hashes ^ h).min() <= near_dup_bits:
            report[ds]["quase_duplicada"] = report[ds].get("quase_duplicada", 0) + 1
            continue
        hashes = np.append(hashes, h)
        items.append({**best, "group": key})
        report[ds]["fotos_de_origem"] = report[ds].get("fotos_de_origem", 0) + 1

    # Limita imagens sem nenhuma caixa (fundo) para não desequilibrar o treino.
    positives = [it for it in items if it["lines"]]
    backgrounds = [it for it in items if not it["lines"]]
    random.shuffle(backgrounds)
    return positives + backgrounds[: int(len(positives) * max_background)], report


def split_items(items, ratios=(0.7, 0.2, 0.1)):
    """Split estratificado pela presença de poste_apagado."""
    strata = defaultdict(list)
    for it in items:
        strata[any(l.startswith("1 ") for l in it["lines"])].append(it)
    splits = {s: [] for s in SPLITS}
    for members in strata.values():
        random.shuffle(members)
        n_train = round(len(members) * ratios[0])
        n_val = round(len(members) * ratios[1])
        splits["train"] += members[:n_train]
        splits["val"] += members[n_train:n_train + n_val]
        splits["test"] += members[n_train + n_val:]
    return splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-background", type=float, default=0.1,
                        help="máximo de imagens sem caixa, como fração das imagens com caixa")
    parser.add_argument("--night-synth", type=float, default=0.5,
                        help="probabilidade de gerar versão noturna de uma foto diurna com poste_apagado (treino)")
    parser.add_argument("--gray-copy", type=float, default=0.3,
                        help="probabilidade de gerar cópia em cinza de uma foto colorida com poste_aceso (treino)")
    parser.add_argument("--near-dup-bits", type=int, default=4,
                        help="distância de Hamming máxima do dHash para considerar quase duplicata")
    parser.add_argument("--out", type=Path, default=DATASET_DIR,
                        help="pasta de saída (outra pasta permite gerar variações para ablação)")
    args = parser.parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    if not RAW_DIR.exists():
        raise SystemExit("Nada em datasets/raw. Rode scripts/download_datasets.py antes.")

    items, report = collect(args.max_background, args.near_dup_bits)
    splits = split_items(items)

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    synth = Counter()
    for split, members in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for i, it in enumerate(members):
            name = f"{it['prefix']}_{i:05d}"
            img = cv2.imread(str(it["img"]))
            cv2.imwrite(str(img_dir / f"{name}.jpg"), img)
            label = "\n".join(it["lines"])
            (lbl_dir / f"{name}.txt").write_text(label)
            if split != "train":
                continue
            classes = {int(l.split()[0]) for l in it["lines"]}
            if 1 in classes and not eh_noturna(img) and rng.random() < args.night_synth:
                cv2.imwrite(str(img_dir / f"{name}_noite.jpg"), day_to_night(img, rng))
                (lbl_dir / f"{name}_noite.txt").write_text(label)
                synth["noite_sintetica"] += 1
            if 0 in classes and not is_gray(img) and rng.random() < args.gray_copy:
                cv2.imwrite(str(img_dir / f"{name}_cinza.jpg"), to_gray(img))
                (lbl_dir / f"{name}_cinza.txt").write_text(label)
                synth["copia_cinza"] += 1

    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASSES)),
    }, sort_keys=False, allow_unicode=True))
    report["_split"] = {s: len(m) for s, m in splits.items()}
    report["_sinteticas_treino"] = dict(synth)
    (out / "merge_report.yaml").write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True))

    for name, info in report.items():
        print(f"\n{name}")
        for k, v in (info.items() if isinstance(info, dict) else []):
            print(f"  {k}: {v}")
    print(f"\ndata.yaml: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
