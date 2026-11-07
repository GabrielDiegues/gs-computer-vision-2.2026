"""Baixa os datasets públicos listados em configs/datasets.yaml (formato YOLO)."""
import os
import sys

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

from common import RAW_DIR, ROOT


def latest_version(project) -> int:
    numbers = [int(str(v.version).rstrip("/").split("/")[-1]) for v in project.versions()]
    if not numbers:
        raise RuntimeError(f"{project.id} não tem versões publicadas")
    return max(numbers)


def main() -> None:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        sys.exit("Defina ROBOFLOW_API_KEY no arquivo .env (veja .env.example).")

    config = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    rf = Roboflow(api_key=api_key)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for ds in config["datasets"]:
        name = f"{ds['workspace']}/{ds['project']}"
        try:
            project = rf.workspace(ds["workspace"]).project(ds["project"])
            version = ds.get("version") or latest_version(project)
            target = RAW_DIR / f"{ds['workspace']}__{ds['project']}__v{version}"
            if target.exists():
                print(f"[ok] {name} v{version} já baixado")
                continue
            print(f"[..] baixando {name} v{version}")
            project.version(version).download("yolov8", location=str(target))
        except Exception as exc:  # um dataset indisponível não deve derrubar os outros
            print(f"[erro] {name}: {exc}")


if __name__ == "__main__":
    main()
