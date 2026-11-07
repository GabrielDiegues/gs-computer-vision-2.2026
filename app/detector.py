"""Carrega o YOLO uma única vez e converte as detecções no veredito do relato."""
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from app.scene import brilho_medio, NIGHT_V_THRESHOLD
from app.schemas import Deteccao, RespostaDeteccao


class Detector:
    def __init__(self, weights: Path, conf: float, verdict_threshold: float):
        self.weights = weights
        self.model = YOLO(str(weights))
        self.conf = conf
        self.verdict_threshold = verdict_threshold
        self.classes = [self.model.names[i] for i in sorted(self.model.names)]

    def detect(self, image: np.ndarray) -> RespostaDeteccao:
        start = time.perf_counter()
        result = self.model.predict(image, conf=self.conf, verbose=False)[0]
        elapsed = (time.perf_counter() - start) * 1000

        deteccoes = [
            Deteccao(classe=self.model.names[int(c)], confianca=round(float(p), 4),
                     bbox=[round(float(v), 1) for v in xyxy])
            for xyxy, p, c in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)
        ]
        brilho = brilho_medio(image)
        noturna = brilho < NIGHT_V_THRESHOLD
        veredito, motivo = self.verdict(deteccoes, noturna)
        h, w = image.shape[:2]
        return RespostaDeteccao(veredito=veredito, motivo=motivo, cena="noite" if noturna else "dia",
                                brilho_medio=round(brilho, 1), deteccoes=deteccoes,
                                largura=w, altura=h, modelo=self.weights.name,
                                tempo_ms=round(elapsed, 1))

    def verdict(self, deteccoes: list[Deteccao], noturna: bool) -> tuple[str, str]:
        """Regra de decisão. A IA sugere; quem decide o envio à Prefeitura é a revisão humana."""
        t = self.verdict_threshold
        if not noturna:
            # De dia o poste fica desligado por projeto: a foto não prova defeito.
            return "inconclusivo", "foto diurna: poste desligado de dia é esperado; reenviar foto à noite"
        apagados = [d for d in deteccoes if d.classe == "poste_apagado" and d.confianca >= t]
        acesos = [d for d in deteccoes if d.classe == "poste_aceso" and d.confianca >= t]
        if apagados:
            return "confirmado", f"{len(apagados)} poste(s) apagado(s) com confiança ≥ {t}"
        if acesos:
            return "nao_confirmado", f"apenas poste(s) aceso(s) detectado(s) com confiança ≥ {t}"
        return "inconclusivo", f"nenhum poste detectado com confiança ≥ {t}; encaminhar para revisão humana"
