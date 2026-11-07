"""Contrato da API de visão consumido pelo fluxo n8n do Rua Acesa."""
from typing import Literal

from pydantic import BaseModel, Field


class Deteccao(BaseModel):
    classe: Literal["poste_aceso", "poste_apagado"]
    confianca: float = Field(ge=0, le=1)
    bbox: list[float] = Field(description="[x1, y1, x2, y2] em pixels da imagem original")


class RespostaDeteccao(BaseModel):
    veredito: Literal["confirmado", "nao_confirmado", "inconclusivo"] = Field(
        description="confirmado: poste apagado detectado; nao_confirmado: só postes acesos; "
                    "inconclusivo: nenhum poste com confiança suficiente (vai para revisão humana)")
    motivo: str
    cena: Literal["dia", "noite"] = Field(description="estimada pelo brilho médio da foto")
    brilho_medio: float = Field(description="média do canal V (HSV), 0–255")
    deteccoes: list[Deteccao]
    largura: int
    altura: int
    modelo: str
    tempo_ms: float


class Saude(BaseModel):
    status: Literal["ok"]
    modelo: str
    classes: list[str]
