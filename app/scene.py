"""Classificação simples da cena (dia/noite) pelo brilho médio da imagem."""
import cv2
import numpy as np

# Média do canal V (HSV) abaixo da qual a foto é considerada noturna.
# Calibrado olhando a distribuição do dataset: fotos noturnas ficam entre ~20 e ~85,
# fotos diurnas acima de ~110.
NIGHT_V_THRESHOLD = 90.0


def brilho_medio(image: np.ndarray) -> float:
    small = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean())


def eh_noturna(image: np.ndarray) -> bool:
    return brilho_medio(image) < NIGHT_V_THRESHOLD
