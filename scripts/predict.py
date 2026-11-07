"""Inferência em imagem, pasta, vídeo ou webcam (--source 0) — usado no vídeo de demonstração."""
import argparse

import cv2
from ultralytics import YOLO

from common import MODELS_DIR, RUNS_DIR

WINDOW = "Rua Acesa - deteccao"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="arquivo, pasta, URL ou índice da webcam (0)")
    parser.add_argument("--weights", default=str(MODELS_DIR / "best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--pausa", type=float, default=3.0,
                        help="segundos que cada imagem fica na tela (0 = espera uma tecla); vídeo roda direto")
    parser.add_argument("--no-show", action="store_true", help="não abrir janela (só salvar)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    model = YOLO(args.weights)
    if not args.no_show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 720)
    # stream=True processa quadro a quadro, sem acumular resultados na memória (vídeo/webcam).
    for result in model.predict(source=source, conf=args.conf, save=True, project=str(RUNS_DIR),
                                name="predict", exist_ok=True, stream=True, verbose=False):
        n_aceso = sum(model.names[int(c)] == "poste_aceso" for c in result.boxes.cls)
        n_apagado = len(result.boxes) - n_aceso
        print(f"{result.path}: {n_aceso} aceso(s), {n_apagado} apagado(s), {result.speed['inference']:.1f} ms")
        if args.no_show:
            continue
        cv2.imshow(WINDOW, result.plot())
        is_image = result.path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        wait = int(args.pausa * 1000) if is_image else 1
        if cv2.waitKey(wait) & 0xFF in (ord("q"), 27):  # q ou Esc encerra
            break
    cv2.destroyAllWindows()
    print(f"Saída salva em {RUNS_DIR / 'predict'}")


if __name__ == "__main__":
    main()
