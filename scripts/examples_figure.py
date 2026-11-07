"""Figura de exemplos para o relatório: detecções + veredito da API para cada imagem de samples/."""
import cv2
import numpy as np

from common import CLASSES, DOCS_IMG, MODELS_DIR, ROOT
from app.detector import Detector

COLORS = {"poste_aceso": (60, 220, 255), "poste_apagado": (255, 120, 60)}
VERDICT_COLORS = {"confirmado": (40, 160, 40), "nao_confirmado": (60, 60, 200), "inconclusivo": (0, 150, 230)}
TILE = 360


def tile(path, detector):
    img = cv2.imread(str(path))
    resp = detector.detect(img)
    for d in resp.deteccoes:
        x0, y0, x1, y1 = map(int, d.bbox)
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[d.classe], 3)
        cv2.putText(img, f"{d.classe.split('_')[1]} {d.confianca:.2f}", (x0, max(18, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[d.classe], 2)
    img = cv2.resize(img, (TILE, TILE))
    banner = np.full((54, TILE, 3), VERDICT_COLORS[resp.veredito], np.uint8)
    cv2.putText(banner, resp.veredito.upper(), (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(banner, f"{path.stem}  cena={resp.cena}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([img, banner])


def main() -> None:
    detector = Detector(MODELS_DIR / "best.pt", conf=0.25, verdict_threshold=0.5)
    names = ["noite_aceso_1", "noite_aceso_2", "noite_aceso_3",
             "noite_sintetica_apagado_1", "noite_sintetica_apagado_2", "noite_sintetica_apagado_3",
             "dia_apagado_1", "dia_apagado_2"]
    tiles = [tile(ROOT / "samples" / f"{n}.jpg", detector) for n in names]
    tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)])
    out = DOCS_IMG / "exemplos_resultados.jpg"
    cv2.imwrite(str(out), grid)
    print(out)


if __name__ == "__main__":
    main()
