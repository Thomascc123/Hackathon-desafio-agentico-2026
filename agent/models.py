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


class ArticleTextResult(BaseModel):
    numero: str = ""
    texto: str = ""
    texto_completo: str = ""
    modificaciones: int = 0
    documento: str = ""
    documento_asunto: str = ""


class KeywordSearchResult(BaseModel):
    articulo: str = ""
    texto: str = ""
    texto_completo: str = ""
    documento_codigo: str = ""
    documento: str = ""
    documento_label: str = ""
    documento_asunto: str = ""
    modificaciones: int = 0


class EvolutionItem(BaseModel):
    modificado_por: str = ""
    fecha: str = ""
    anio: str = ""
    accion: str = ""
    articulo: str = ""
    texto_actual: str = ""
    num_modificaciones: int = 0


class DocumentTimelineResult(BaseModel):
    id: str = ""
    numero: str = ""
    fecha: str = ""
    anio: str = ""
    resuelve: str = ""
    autoridad: str = ""


class DocumentSearchResult(BaseModel):
    id: str = ""
    numero: str = ""
    fecha: str = ""
    resuelve: str = ""
    autoridad: str = ""


class HelpResult(BaseModel):
    help_text: str = ""


class AgentResponse(BaseModel):
    answer: str = Field(description="Complete answer in Spanish with citations")
    sources: list[str] = Field(description="List of cited sources")
    disclaimer: str = Field(default="Información basada en documentos oficiales de normativa.udea.edu.co")
