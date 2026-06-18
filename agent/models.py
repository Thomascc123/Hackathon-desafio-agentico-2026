from dataclasses import dataclass
from typing import Any
from pydantic import BaseModel, Field


@dataclass
class AgentDeps:
    graph: Any
    chroma_collection: Any
    pdf_dir: str
    audit: Any


class ArticleResult(BaseModel):
    numero: str = Field(description="Article number")
    texto: str = Field(description="Article text")
    documento: str = Field(default="", description="Source document label")
    score: float = Field(default=0.0, description="Relevance score")


class ModificationEvent(BaseModel):
    anio: str = Field(description="Year of modification")
    modificado_por: str = Field(description="Document that modified")
    accion: str = Field(default="", description="Description of action")


class ArticleHistory(BaseModel):
    articulo: str = Field(description="Article number")
    texto_actual: str = Field(description="Current text of the article")
    num_modificaciones: int = Field(default=0)
    modificaciones: list[ModificationEvent] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    numero: str = Field(description="Document number")
    fecha: str = Field(description="Date")
    resuelve: str = Field(description="Summary of what the document does")
    autoridad: str = Field(default="", description="Issuing authority")
    anio: str = Field(default="", description="Year")


class AgentResponse(BaseModel):
    answer: str = Field(description="Complete answer in Spanish with citations")
    sources: list[str] = Field(description="List of cited sources")
    disclaimer: str = Field(default="Información basada en documentos oficiales de normativa.udea.edu.co")
