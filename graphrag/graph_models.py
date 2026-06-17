from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class NodeType(str, Enum):
    DOCUMENTO = "Documento"
    ARTICULO = "Articulo"
    CAPITULO = "Capitulo"
    TITULO = "Titulo"
    CONCEPTO = "Concepto"
    AUTORIDAD = "Autoridad"


class EdgeType(str, Enum):
    CONTIENE = "CONTIENE"           # Documento -> Capitulo/Titulo/Articulo
    MODIFICA = "MODIFICA"           # Documento -> Articulo (modifying doc modifies article)
    DEROGA = "DEROGA"               # Documento -> Documento
    CITA = "CITA"                   # Documento -> Documento
    REGLAMENTA = "REGLAMENTA"       # Documento -> Concepto
    ACTUALIZA = "ACTUALIZA"         # Documento -> Documento (version actualizada)
    EMITE = "EMITE"                 # Autoridad -> Documento


@dataclass
class Node:
    id: str
    type: NodeType
    label: str
    properties: dict = field(default_factory=dict)


@dataclass
class Edge:
    source_id: str
    target_id: str
    type: EdgeType
    properties: dict = field(default_factory=dict)


@dataclass
class DocumentMetadata:
    codigo: str
    numero: str
    fecha: str
    vigencia: str
    medio: str
    resuelve: str
    normas_relacionadas: str
    tipo_documento: str = ""
    autoridad: str = ""
    asunto: str = ""


@dataclass
class ParsedDocument:
    metadata: DocumentMetadata
    capitulos: list = field(default_factory=list)
    articulos: list = field(default_factory=list)
    texto_completo: str = ""
