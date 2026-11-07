"""Gera o vídeo de demonstração (sem áudio) a partir da execução real da solução.

- Detecção: o modelo roda de verdade em cada quadro de uma "câmera" em movimento
  (recorte com zoom e deslocamento sobre fotos do conjunto de teste).
- API: sobe o servidor FastAPI, faz requisições HTTP reais com curl e mostra as
  respostas devolvidas.
- Cartelas com legendas no lugar da narração.

Pré-requisito: `python evaluate.py --cenario noite_sintetica` (gera as noites sintéticas do teste).
Uso: python make_demo_video.py  ->  docs/video/demo_rua_acesa.mp4
"""
import json
import math
import random
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import DATASET_DIR, DOCS_IMG, MODELS_DIR, ROOT
from app.detector import Detector

W, H, FPS = 1920, 1080, 30
FADE = 8  # quadros de fade em cada ponta dos segmentos
OUT = ROOT / "docs" / "video" / "demo_rua_acesa.mp4"
REPO = "github.com/GabrielDiegues/gs-computer-vision-2.2026"
TEAM = ["Gabriel Diegues Figueiredo Rocha — RM 55078", "Pedro Henrique Palladino — RM 551180",
        "Pedro Henrique Pontes — RM 98036", "Raí Gumieri dos Santos — RM 98287"]

BG, PANEL, TERM = (15, 20, 33), (28, 35, 54), (8, 11, 20)
FG, MUTED, ACCENT = (235, 238, 245), (150, 160, 180), (250, 204, 21)
CLS_COLOR = {"poste_aceso": (250, 204, 21), "poste_apagado": (34, 211, 238)}
VERDICT_COLOR = {"confirmado": (220, 53, 69), "nao_confirmado": (40, 167, 69), "inconclusivo": (245, 158, 11)}
VERDICT_LABEL = {"confirmado": "CONFIRMADO", "nao_confirmado": "NÃO CONFIRMADO", "inconclusivo": "INCONCLUSIVO"}

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size: int, bold: bool = False, mono: bool = False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    path = FONT_DIR / f"{name}{'-Bold' if bold else ''}.ttf"
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def br(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


class Video:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def emit(self, frames, total: int):
        """Escreve `total` quadros (iterável de arrays RGB) com fade de entrada e saída."""
        for i, f in enumerate(frames):
            a = min(1.0, (i + 1) / FADE, (total - i) / FADE)
            if a < 1.0:
                f = (f.astype(np.float32) * a).astype(np.uint8)
            self.proc.stdin.write(np.ascontiguousarray(f).tobytes())
            self.frames += 1

    def hold(self, frame: np.ndarray, seconds: float):
        n = int(seconds * FPS)
        self.emit((frame for _ in range(n)), n)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------- desenho ----------

def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap(d, text, fnt, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if d.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    return lines + [cur] if cur else lines


def paragraph(d, xy, text, fnt, width, fill=FG, spacing=1.35):
    x, y = xy
    for line in wrap(d, text, fnt, width):
        d.text((x, y), line, font=fnt, fill=fill)
        y += int(fnt.size * spacing)
    return y


def header(d, title, subtitle=None):
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.text((60, 34), title, font=font(46, bold=True), fill=FG)
    if subtitle:
        d.text((60, 96), subtitle, font=font(28), fill=MUTED)


def footer(d, text):
    d.text((60, H - 48), text, font=font(22), fill=MUTED)


def paste(img, arr_bgr, box):
    """Cola uma imagem OpenCV (BGR) redimensionada para caber em box=(x, y, w, h), centralizada."""
    x, y, w, h = box
    ih, iw = arr_bgr.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(iw * s), int(ih * s)
    im = Image.fromarray(cv2.cvtColor(cv2.resize(arr_bgr, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    img.paste(im, (ox, oy))
    return ox, oy, s, ox + nw


def draw_boxes(d, deteccoes, ox, oy, s, right=W):
    f = font(22, bold=True)
    for det in deteccoes:
        x1, y1, x2, y2 = [v * s for v in det["bbox"]]
        x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
        color = CLS_COLOR[det["classe"]]
        d.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{'aceso' if det['classe'] == 'poste_aceso' else 'apagado'} {br(det['confianca'])}"
        tw = d.textlength(label, font=f)
        ty = y1 - 30 if y1 - 30 > oy else y1
        lx = min(x1, right - tw - 12)
        d.rectangle([lx, ty, lx + tw + 12, ty + 30], fill=color)
        d.text((lx + 6, ty + 3), label, font=f, fill=(10, 10, 10))


def badge(d, xy, veredito, size=40):
    f = font(size, bold=True)
    label = VERDICT_LABEL[veredito]
    x, y = xy
    tw = d.textlength(label, font=f)
    d.rounded_rectangle([x, y, x + tw + 48, y + size + 30], radius=14, fill=VERDICT_COLOR[veredito])
    d.text((x + 24, y + 13), label, font=f, fill=(255, 255, 255))
    return y + size + 30


def arr(img):
    return np.asarray(img)


# ---------- cartelas ----------

def title_card():
    img, d = canvas()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.text((140, 250), "Rua Acesa — Visão", font=font(92, bold=True), fill=FG)
    d.text((140, 375), "Detecção de postes acesos e apagados com YOLO26 treinado sob medida",
           font=font(40), fill=ACCENT)
    d.text((140, 450), "Global Solution 2026.2 · CitizenTech · Applied Computer Vision · FIAP",
           font=font(30), fill=MUTED)
    y = 590
    for name in TEAM:
        d.text((140, y), name, font=font(32), fill=FG)
        y += 52
    d.text((140, H - 110), "Demonstração da solução funcionando (vídeo sem áudio, com legendas)",
           font=font(26), fill=MUTED)
    return arr(img)


def pipeline_card():
    img, d = canvas()
    header(d, "Onde a visão entra no Rua Acesa",
           "Ruas com poste apagado deixam o trajeto noturno inseguro. O cidadão envia a foto; a IA confirma.")
    nodes = ["Cidadão ou robô\nenvia a foto", "API central\ndo produto", "n8n (RPA)\nvalida o relato",
             "YOLO26\nPOST /detect", "Veredito\n+ revisão humana"]
    bw, bh, gap, y = 300, 150, 60, 300
    x0 = (W - (len(nodes) * bw + (len(nodes) - 1) * gap)) // 2
    for i, label in enumerate(nodes):
        x = x0 + i * (bw + gap)
        hl = i == 3
        d.rounded_rectangle([x, y, x + bw, y + bh], radius=18, fill=ACCENT if hl else PANEL,
                            outline=ACCENT, width=3)
        for j, line in enumerate(label.split("\n")):
            f = font(30 if j == 0 else 24, bold=j == 0)
            tw = d.textlength(line, font=f)
            d.text((x + (bw - tw) / 2, y + 38 + j * 46), line, font=f, fill=(15, 15, 15) if hl else FG)
        if i < len(nodes) - 1:
            ax = x + bw + 8
            d.line([ax, y + bh / 2, ax + gap - 16, y + bh / 2], fill=MUTED, width=4)
            d.polygon([(ax + gap - 16, y + bh / 2 - 10), (ax + gap - 4, y + bh / 2), (ax + gap - 16, y + bh / 2 + 10)], fill=MUTED)
    tx = x0 + 3 * (bw + gap)
    d.text((tx - 20, y + bh + 20), "← este repositório", font=font(26, bold=True), fill=ACCENT)
    rows = [("confirmado", "foto noturna com poste apagado (confiança ≥ 0,5) → segue para a fila de revisão"),
            ("nao_confirmado", "foto noturna só com postes acesos → relato arquivado com justificativa"),
            ("inconclusivo", "nenhum poste confiável ou foto diurna → revisão humana / pedir nova foto à noite")]
    y = 610
    for v, text in rows:
        badge(d, (160, y), v, size=28)
        d.text((560, y + 14), text, font=font(28), fill=FG)
        y += 100
    footer(d, "IA responsável: o modelo sugere, uma pessoa revisa antes de qualquer envio à Prefeitura. A imagem não é armazenada (LGPD).")
    return arr(img)


def data_card():
    img, d = canvas()
    header(d, "Dados e treinamento", "Somente datasets públicos (Roboflow Universe, CC BY 4.0)")
    sample = cv2.imread(str(DOCS_IMG / "amostras_dataset.jpg"))
    paste(img, sample, (60, 150, 840, 840))
    d.text((60, 1000), "Amostras do treino (amarelo: aceso · azul: apagado)", font=font(22), fill=MUTED)
    bullets = [
        ("5 datasets públicos unificados", "classes originais remapeadas para poste_aceso e poste_apagado"),
        ("~15 mil arquivos → 2.814 fotos de origem", "deduplicação exata (MD5) e perceptual (dHash): sem vazamento entre treino e teste"),
        ("Correção de atalho: noite sintética", "quase todo poste apagado público era foto de dia; o treino ganhou versões escurecidas para o modelo olhar a lâmpada, não o céu"),
        ("YOLO26s com fine-tuning (COCO)", "47,6 min numa RTX 4060 · split 70/20/10 · teste nunca visto no treino"),
    ]
    y = 170
    for title, body in bullets:
        d.ellipse([960, y + 12, 976, y + 28], fill=ACCENT)
        d.text((1000, y), title, font=font(32, bold=True), fill=FG)
        y = paragraph(d, (1000, y + 48), body, font(26), 840, fill=MUTED) + 34
    return arr(img)


def section_card(num, title, subtitle):
    img, d = canvas()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.text((140, 380), f"{num}", font=font(120, bold=True), fill=ACCENT)
    d.text((340, 400), title, font=font(60, bold=True), fill=FG)
    paragraph(d, (340, 500), subtitle, font(32), 1400, fill=MUTED)
    return arr(img)


def results_card():
    img, d = canvas()
    header(d, "Resultados no conjunto de teste", "281 imagens nunca vistas no treino · meta mAP50 ≥ 0,6 superada em todos os cenários")

    def table(x, y, cols, widths, rows, bold_col=None):
        fh, fb = font(26, bold=True), font(26)
        d.rectangle([x, y, x + sum(widths), y + 52], fill=PANEL)
        cx = x
        for c, w in zip(cols, widths):
            d.text((cx + 16, y + 12), c, font=fh, fill=ACCENT)
            cx += w
        y += 52
        for r in rows:
            cx = x
            for i, (c, w) in enumerate(zip(r, widths)):
                d.text((cx + 16, y + 12), c, font=fh if i == bold_col else fb, fill=FG)
                cx += w
            d.line([x, y + 52, x + sum(widths), y + 52], fill=PANEL, width=2)
            y += 52
        return y

    y = table(60, 160, ["Cenário", "Imagens", "mAP50", "mAP50 apagado", "Recall apagado"],
              [400, 150, 150, 250, 250],
              [["Completo", "281", "0,766", "0,840", "0,838"],
               ["Noite (fotos reais)", "47", "0,823", "0,898", "0,867"],
               ["Noite simulada", "234", "0,734", "0,831", "0,847"]], bold_col=3)
    d.text((60, y + 40), "Ablação: o mesmo treino sem a noite sintética", font=font(34, bold=True), fill=FG)
    y = table(60, y + 100, ["Cenário", "mAP50 apagado", "Recall apagado"], [400, 400, 400],
              [["Completo", "0,840  vs  0,834", "0,838  vs  0,790"],
               ["Noite simulada", "0,831  vs  0,433", "0,847  vs  0,498"]])
    paragraph(d, (60, y + 30), "Sem a correção, o modelo encontra só metade dos postes apagados à noite: tinha aprendido "
              "\"céu claro = apagado\". Inferência: cerca de 6 ms por imagem na GPU.", font(28), 1150, fill=MUTED)
    cm = cv2.imread(str(DOCS_IMG / "test_completo_confusion_matrix_normalized.png"))
    if cm is not None:
        paste(img, cm, (1300, 150, 570, 480))
        paragraph(d, (1310, 650), "Matriz de confusão (teste completo): só 2% dos postes apagados viraram "
                  "\"aceso\" e nenhum aceso virou \"apagado\".", font(22), 550, fill=MUTED)
    return arr(img)


def closing_card():
    img, d = canvas()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    d.text((140, 260), "Rua Acesa — Visão", font=font(72, bold=True), fill=FG)
    items = ["YOLO26s treinado sob medida: poste aceso e poste apagado",
             "API FastAPI com veredito pronto para o fluxo n8n",
             "Guarda dia/noite e revisão humana obrigatória",
             "Imagem processada em memória, nunca armazenada (LGPD)"]
    y = 400
    for it in items:
        d.ellipse([140, y + 12, 156, y + 28], fill=ACCENT)
        d.text((180, y), it, font=font(36), fill=FG)
        y += 70
    d.text((140, 760), REPO, font=font(36, bold=True), fill=ACCENT)
    return arr(img)


# ---------- detecção quadro a quadro ----------

def detection_clip(video: Video, det: Detector, img_bgr, title, subtitle, caption, seconds, seed):
    rng = random.Random(seed)
    ih, iw = img_bgr.shape[:2]
    zoom_end = rng.uniform(0.80, 0.88)
    c0 = (0.5 + rng.uniform(-0.03, 0.03), 0.5 + rng.uniform(-0.03, 0.03))
    c1 = (0.5 + rng.uniform(-0.06, 0.06), 0.5 + rng.uniform(-0.06, 0.06))
    n = int(seconds * FPS)
    times = []

    def frames():
        for i in range(n):
            t = i / max(1, n - 1)
            e = 0.5 - 0.5 * math.cos(math.pi * t)
            z = 1.0 + (zoom_end - 1.0) * e
            cw, ch = iw * z, ih * z
            cx = (c0[0] + (c1[0] - c0[0]) * e) * iw
            cy = (c0[1] + (c1[1] - c0[1]) * e) * ih
            x0 = int(min(max(cx - cw / 2, 0), iw - cw))
            y0 = int(min(max(cy - ch / 2, 0), ih - ch))
            crop = img_bgr[y0:y0 + int(ch), x0:x0 + int(cw)]
            r = det.detect(crop).model_dump()
            times.append(r["tempo_ms"])

            img, d = canvas()
            header(d, title, subtitle)
            ox, oy, s, right = paste(img, crop, (60, 150, 880, 880))
            draw_boxes(d, r["deteccoes"], ox, oy, s, right)
            d.rectangle([60, 150, 940, 1030], outline=PANEL, width=3)

            x, y = 1010, 160
            d.text((x, y), "Veredito da regra de decisão", font=font(26), fill=MUTED)
            y = badge(d, (x, y + 44), r["veredito"]) + 18
            y = paragraph(d, (x, y), r["motivo"], font(26), 840, fill=FG) + 26
            d.text((x, y), f"Cena: {r['cena']}  ·  brilho médio {br(r['brilho_medio'], 1)}", font=font(28), fill=FG)
            y += 70
            d.text((x, y), "Detecções", font=font(26), fill=MUTED)
            y += 44
            for k in r["deteccoes"][:7]:
                d.rectangle([x, y + 6, x + 22, y + 28], fill=CLS_COLOR[k["classe"]])
                d.text((x + 38, y), f"{k['classe']}   {br(k['confianca'])}", font=font(28, mono=True), fill=FG)
                y += 44
            if not r["deteccoes"]:
                d.text((x, y), "nenhuma", font=font(28), fill=FG)
            avg = sum(times[-15:]) / len(times[-15:])
            d.text((x, 960), f"Inferência real neste quadro: {br(r['tempo_ms'], 1)} ms  (média {br(avg, 1)} ms)",
                   font=font(24), fill=MUTED)
            d.text((x, 994), f"quadro {i + 1}/{n} · cada quadro passa pelo modelo", font=font(22), fill=MUTED)
            footer(d, caption)
            yield arr(img)

    video.emit(frames(), n)


# ---------- API ----------

def start_api(port=8000):
    proc = subprocess.Popen([str(Path(sys.executable)), "-m", "uvicorn", "app.main:app", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1)
            return proc
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise SystemExit("A API não subiu na porta 8000 (já existe outro servidor rodando?).")


def curl(args):
    """Executa o curl de verdade e devolve (status HTTP, corpo)."""
    out = subprocess.run(["curl", "-s", "-o", "-", "-w", "\n%{http_code}", *args], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    body, status = out.rsplit("\n", 1)
    return int(status), body


def format_json(body: str) -> list[str]:
    """JSON indentado, com cada detecção em uma linha (como no README)."""
    data = json.loads(body)
    lines = ["{"]
    items = list(data.items())
    for i, (k, v) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        if k == "deteccoes" and v:
            lines.append(f'  "{k}": [')
            for j, det in enumerate(v):
                lines.append("    " + json.dumps(det, ensure_ascii=False) + ("," if j < len(v) - 1 else ""))
            lines.append(f"  ]{comma}")
        else:
            lines.append(f'  "{k}": {json.dumps(v, ensure_ascii=False)}{comma}')
    lines.append("}")
    return lines


def api_clip(video: Video, command: str, status: int, body: str, image_path=None, note=""):
    lines = format_json(body)
    data = json.loads(body)
    type_s, hold_s = 1.6, 6.5 if image_path else 4.5
    n_type, n = int(type_s * FPS), int((type_s + hold_s) * FPS)
    img_bgr = cv2.imread(str(ROOT / image_path)) if image_path else None
    fm = font(18, mono=True)
    term = (800, 150, 1860, 1030) if image_path else (60, 150, 1860, 1030)

    def frames():
        for i in range(n):
            img, d = canvas()
            header(d, "API FastAPI — o contrato que o n8n consome",
                   "Servidor rodando localmente; requisições HTTP reais feitas com curl")
            if img_bgr is not None:
                ox, oy, s, right = paste(img, img_bgr, (60, 150, 700, 700))
                if i >= n_type:
                    draw_boxes(d, data.get("deteccoes", []), ox, oy, s, right)
                d.text((60, 870), "Imagem enviada:", font=font(24), fill=MUTED)
                d.text((60, 904), image_path, font=font(24), fill=FG)
                if i >= n_type and "veredito" in data:
                    badge(d, (60, 960), data["veredito"], size=30)
            x0, y0, x1, y1 = term
            d.rounded_rectangle(term, radius=12, fill=TERM, outline=PANEL, width=2)
            for k, c in enumerate([(239, 68, 68), (245, 158, 11), (34, 197, 94)]):
                d.ellipse([x0 + 20 + k * 30, y0 + 16, x0 + 36 + k * 30, y0 + 32], fill=c)
            typed = command[: int(len(command) * min(1.0, i / max(1, n_type - 6)))]
            y = y0 + 56
            for ln in textwrap.wrap("$ " + typed, width=int((x1 - x0 - 40) / (fm.size * 0.6))) or ["$ "]:
                d.text((x0 + 20, y), ln, font=fm, fill=(134, 239, 172))
                y += 28
            if i >= n_type:
                y += 10
                d.text((x0 + 20, y), f"HTTP {status}", font=font(18, bold=True, mono=True),
                       fill=(134, 239, 172) if status == 200 else (252, 165, 165))
                y += 34
                for ln in lines:
                    d.text((x0 + 20, y), ln, font=fm, fill=FG)
                    y += 26
            if note:
                footer(d, note)
            yield arr(img)

    video.emit(frames(), n)


# ---------- roteiro ----------

def main() -> None:
    synth_dir = DATASET_DIR / "test_noite_sintetica" / "images"
    test_dir = DATASET_DIR / "images" / "test"
    if not synth_dir.exists():
        raise SystemExit("Rode antes: python evaluate.py --cenario noite_sintetica")

    det = Detector(MODELS_DIR / "best.pt", conf=0.25, verdict_threshold=0.5)
    det.detect(np.zeros((640, 640, 3), np.uint8))  # aquece a GPU

    video = Video(OUT)
    video.hold(title_card(), 6)
    video.hold(pipeline_card(), 10)
    video.hold(data_card(), 11)

    src = "Fonte: conjunto de teste, nunca usado no treino."
    noite_real = ["street_light_on_00046.jpg", "street_light_on_00051.jpg",
                  "street_light_on_00123.jpg", "streetlighttest_00037.jpg"]
    noite_sint = ["pole-p1ls9_00137.jpg", "pole-p1ls9_00227.jpg",
                  "pole-p1ls9_00228.jpg", "streetlighttest_00135.jpg"]
    dia = ["streetlighttest_00135.jpg", "pole-p1ls9_00227.jpg"]

    video.hold(section_card(1, "Postes acesos à noite",
                            "Fotos noturnas reais. O modelo detecta as lâmpadas acesas e o relato de "
                            "\"poste apagado\" não se confirma."), 3.5)
    for k, name in enumerate(noite_real):
        detection_clip(video, det, cv2.imread(str(test_dir / name)), "Detecção quadro a quadro · noite real",
                       "Câmera simulada: recorte em movimento sobre a foto", src, 4, seed=k)

    video.hold(section_card(2, "Postes apagados à noite",
                            "Quase não existem fotos públicas reais deste caso. Usamos fotos diurnas do teste "
                            "escurecidas pela mesma transformação do treino (\"noite simulada\")."), 4.5)
    for k, name in enumerate(noite_sint):
        detection_clip(video, det, cv2.imread(str(synth_dir / name)), "Detecção quadro a quadro · noite simulada",
                       "A lâmpada apagada continua sem brilho próprio: o modelo precisa olhar a luminária",
                       src + " Noite simulada a partir da foto diurna.", 4, seed=10 + k)

    video.hold(section_card(3, "A mesma foto, de dia",
                            "De dia o poste fica desligado por projeto. O modelo vê o poste apagado, mas a "
                            "guarda dia/noite devolve \"inconclusivo\" e pede uma foto noturna."), 4)
    for k, name in enumerate(dia):
        detection_clip(video, det, cv2.imread(str(test_dir / name)), "Detecção quadro a quadro · foto diurna",
                       "Guarda de cena: brilho médio alto = dia", src, 4, seed=20 + k)

    video.hold(section_card(4, "API para o fluxo n8n",
                            "POST /detect recebe a foto do relato e devolve o veredito em JSON."), 3.5)
    api = start_api()
    try:
        curl(["-F", "file=@samples/noite_aceso_2.jpg", "http://localhost:8000/detect"])  # aquece (fora do vídeo)
        status, body = curl(["http://localhost:8000/health"])
        api_clip(video, "curl http://localhost:8000/health", status, body,
                 note="GET /health: status do serviço, modelo carregado e classes.")
        for path, note in [("samples/noite_sintetica_apagado_2.jpg", "Poste apagado à noite → confirmado: segue para deduplicação, prioridade e fila de revisão."),
                           ("samples/noite_aceso_1.jpg", "Só postes acesos → não confirmado: o relato é arquivado com justificativa."),
                           ("samples/dia_apagado_1.jpg", "Foto diurna → inconclusivo: o n8n pede ao cidadão uma nova foto à noite.")]:
            status, body = curl(["-F", f"file=@{path}", "http://localhost:8000/detect"])
            api_clip(video, f'curl -F "file=@{path}" http://localhost:8000/detect', status, body, path, note)
        status, body = curl(["-F", "file=@README.md;type=text/plain", "http://localhost:8000/detect"])
        api_clip(video, 'curl -F "file=@README.md;type=text/plain" http://localhost:8000/detect', status, body,
                 note="Validação: arquivo que não é imagem é recusado (415). A imagem nunca é gravada em disco.")
    finally:
        api.terminate()
        api.wait()

    video.hold(results_card(), 13)
    video.hold(closing_card(), 7)
    video.close()
    print(f"{OUT}  ({video.frames / FPS:.0f} s)")


if __name__ == "__main__":
    main()
