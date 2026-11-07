"""Fine-tuning do YOLO pré-treinado (COCO) no dataset do Rua Acesa."""
import argparse
import shutil

from ultralytics import YOLO

from common import DATA_YAML, MODELS_DIR, RUNS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s.pt", help="pesos pré-treinados de partida")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=0.7,
                        help="fração da VRAM (AutoBatch) ou inteiro com o tamanho do lote")
    parser.add_argument("--name", default="rua_acesa")
    parser.add_argument("--data", default=str(DATA_YAML), help="data.yaml do dataset")
    parser.add_argument("--no-export", action="store_true",
                        help="não copiar o best.pt para models/ (ex.: treinos de ablação)")
    args = parser.parse_args()

    # Pesos pré-treinados ficam em models/ (baixados automaticamente na primeira execução).
    model = YOLO(str(MODELS_DIR / args.model))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch if args.batch < 1 else int(args.batch),
        patience=20,          # early stopping
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        seed=42,
        cos_lr=True,
        # Cenas noturnas: variar brilho/saturação ajuda com câmeras de celular
        # diferentes e com o contraste forte entre lâmpada e céu escuro.
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        plots=True,
    )

    if args.no_export:
        return
    best = RUNS_DIR / args.name / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, MODELS_DIR / "best.pt")
    print(f"Pesos finais copiados para {MODELS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
