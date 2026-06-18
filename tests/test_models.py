from agent.models import (
    ArticleTextResult,
    KeywordSearchResult,
    EvolutionItem,
    DocumentTimelineResult,
    DocumentSearchResult,
    HelpResult,
    AgentResponse,
)


def test_article_text_result_defaults():
    r = ArticleTextResult(numero="63", texto="Texto del artículo")
    assert r.numero == "63"
    assert r.texto == "Texto del artículo"
    assert r.texto_completo == ""
    assert r.modificaciones == 0
    assert r.documento == ""
    assert r.documento_asunto == ""


def test_article_text_result_full():
    r = ArticleTextResult(
        numero="10",
        texto="Resumen",
        texto_completo="Texto completo del artículo 10",
        modificaciones=3,
        documento="REGLAMENTO ESTUDIANTIL DE PREGRADO",
        documento_asunto="REGLAMENTO ESTUDIANTIL DE PREGRADO",
    )
    assert r.model_dump()["texto_completo"] == "Texto completo del artículo 10"
    assert r.modificaciones == 3


def test_keyword_search_result():
    r = KeywordSearchResult(
        articulo="50",
        texto="Texto del artículo 50",
        documento_codigo="12345",
        documento="AS 999/2024",
        documento_label="AS 999/2024",
        documento_asunto="MATRICULA",
        modificaciones=1,
    )
    assert r.articulo == "50"
    assert r.documento_label == "AS 999/2024"
    assert r.documento_asunto == "MATRICULA"


def test_evolution_item():
    r = EvolutionItem(
        modificado_por="AS 458/2019",
        fecha="2019-06-15",
        anio="2019",
        accion="Modifica el artículo 63",
    )
    assert r.modificado_por == "AS 458/2019"
    assert r.model_dump()["accion"] == "Modifica el artículo 63"
    # Fields that should be empty by default
    assert r.articulo == ""
    assert r.texto_actual == ""
    assert r.num_modificaciones == 0


def test_document_timeline_result():
    r = DocumentTimelineResult(
        id="DOC:35137059",
        numero="623",
        fecha="2025-06-15",
        anio="2025",
        resuelve="Modificar el Reglamento Estudiantil de Pregrado",
        autoridad="Consejo Superior Universitario",
    )
    assert r.numero == "623"
    assert r.anio == "2025"


def test_document_search_result():
    r = DocumentSearchResult(
        id="DOC:35137059",
        numero="623",
        fecha="2025-06-15",
        resuelve="Modificación del reglamento",
        autoridad="CSU",
    )
    assert r.id == "DOC:35137059"
    assert r.numero == "623"


def test_help_result():
    r = HelpResult(help_text="**Comandos disponibles:**\n- ¿Qué dice el artículo X?")
    assert "Comandos disponibles" in r.help_text


def test_agent_response():
    r = AgentResponse(
        answer="Según el Artículo 63...",
        sources=["Art. 63", "AS 458/2019"],
    )
    assert len(r.sources) == 2
    assert r.disclaimer == "Información basada en documentos oficiales de normativa.udea.edu.co"
