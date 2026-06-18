import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphrag.graph_builder import NetworkXNormativaGraph
from graphrag.graph_models import NodeType, EdgeType
from graphrag.query_agent import GraphRAGAgent


@pytest.fixture
def graph():
    g = NetworkXNormativaGraph()
    doc_id = g.add_document(
        codigo="35137059",
        numero="623",
        fecha="2025-06-15",
        resuelve="Modificar el Reglamento Estudiantil de Pregrado",
        tipo_doc="AS",
        asunto="REGLAMENTO ESTUDIANTIL DE PREGRADO",
    )
    art_id = g.add_articulo(
        doc_codigo="35137059",
        art_numero="63",
        texto="Ningún estudiante podrá matricularse simultáneamente en más de un programa académico.",
        modificaciones=[{"tipo": "AS", "numero": "623"}],
    )
    g.add_contiene(doc_id, art_id, orden=63)
    return g


@pytest.fixture
def agent(graph):
    return GraphRAGAgent(graph)


@pytest.fixture
def graph_with_modifications():
    g = NetworkXNormativaGraph()
    doc_id = g.add_document(
        codigo="35137054",
        numero="458",
        fecha="2019-06-15",
        resuelve="Modificar el artículo 63 del Reglamento",
        tipo_doc="AS",
        asunto="REGLAMENTO ESTUDIANTIL DE PREGRADO",
    )
    doc2_id = g.add_document(
        codigo="35137059",
        numero="623",
        fecha="2025-06-15",
        resuelve="Modificar el Reglamento Estudiantil de Pregrado",
        tipo_doc="AS",
        asunto="REGLAMENTO ESTUDIANTIL DE PREGRADO",
    )
    art_id = g.add_articulo(
        doc_codigo="35137054",
        art_numero="63",
        texto="Texto original del artículo 63",
        modificaciones=[{"tipo": "AS", "numero": "458"}, {"tipo": "AS", "numero": "623"}],
    )
    g.add_contiene(doc_id, art_id, orden=63)
    g.add_modifica(doc2_id, art_id, "63")
    return g


# ── classify_intent ────────────────────────────────────────────────

class TestClassifyIntent:
    def test_article_text_dice(self, agent):
        intent, params = agent.classify_intent("qué dice el artículo 63")
        assert intent == "article_text"
        assert params.get("art_num") == "63"

    def test_article_text_simple(self, agent):
        intent, params = agent.classify_intent("artículo 50")
        assert intent == "article_text"
        assert params.get("art_num") == "50"

    def test_article_text_show(self, agent):
        intent, params = agent.classify_intent("muestra el artículo 15")
        assert intent == "article_text"
        assert params.get("art_num") == "15"

    def test_evolution(self, agent):
        intent, params = agent.classify_intent("historia del artículo 130")
        assert intent == "evolution"
        assert params.get("art_num") == "130"

    def test_modified_by(self, agent):
        intent, params = agent.classify_intent("qué acuerdos modifican el artículo 63")
        assert intent == "modified_by"
        assert params.get("art_num") == "63"

    def test_keyword_search(self, agent):
        intent, params = agent.classify_intent("artículos sobre matrícula")
        assert intent == "keyword_search"
        assert "matrícula" in params.get("keyword", "")

    def test_document_timeline(self, agent):
        intent, params = agent.classify_intent("documentos de pregrado")
        assert intent == "document_timeline"
        assert params.get("asunto", "").lower() == "pregrado"

    def test_document_timeline_acuerdos(self, agent):
        intent, params = agent.classify_intent("acuerdos de pregrado")
        assert intent == "document_timeline"

    def test_concept_query(self, agent):
        intent, params = agent.classify_intent("qué son las matrículas de honor")
        assert intent == "concept_query"
        assert "matrículas de honor" in params.get("concept", "").lower()

    def test_document_search(self, agent):
        intent, params = agent.classify_intent("busca el acuerdo superior 458 de 2019")
        assert intent == "document_search"
        assert params.get("doc_num") == "458"
        assert params.get("doc_year") == "2019"

    def test_help(self, agent):
        intent, params = agent.classify_intent("ayuda")
        assert intent == "help"

    def test_unknown_intent(self, agent):
        intent, params = agent.classify_intent("cómo está el clima hoy")
        assert intent == "unknown"


# ── Handlers ───────────────────────────────────────────────────────

class TestHandlers:
    def test_handle_article_text_found(self, agent):
        results = agent._handle_article_text({"art_num": "63"})
        assert len(results) == 1
        assert results[0].numero == "63"
        assert "matricularse" in results[0].texto

    def test_handle_article_text_not_found(self, agent):
        results = agent._handle_article_text({"art_num": "999"})
        assert len(results) == 0

    def test_handle_keyword_search(self, agent):
        results = agent._handle_keyword_search({"keyword": "matricularse"})
        assert len(results) >= 1
        assert results[0].articulo == "63"

    def test_handle_keyword_search_no_results(self, agent):
        results = agent._handle_keyword_search({"keyword": "zzznotfound"})
        assert len(results) == 0

    def test_handle_help(self, agent):
        results = agent._handle_help({})
        assert len(results) == 1
        assert "Comandos disponibles" in results[0].help_text

    def test_handle_document_search(self, agent):
        results = agent._handle_document_search({"doc_num": "623", "doc_year": ""})
        assert len(results) >= 1
        assert results[0].numero == "623"

    def test_handle_evolution(self, graph_with_modifications):
        agent = GraphRAGAgent(graph_with_modifications)
        results = agent._handle_evolution({"art_num": "63"})
        assert len(results) >= 2  # at least one mod event + one current text
        mods = [r for r in results if r.modificado_por]
        assert len(mods) >= 1

    def test_handle_modified_by(self, graph_with_modifications):
        agent = GraphRAGAgent(graph_with_modifications)
        results = agent._handle_modified_by({"art_num": "63"})
        assert len(results) >= 1
        assert results[0].modificado_por != ""


# ── answer() integration ───────────────────────────────────────────

class TestAnswer:
    def test_article_text(self, agent):
        answer = agent.answer("qué dice el artículo 63")
        assert "Artículo 63" in answer
        assert "matricularse" in answer

    def test_help(self, agent):
        answer = agent.answer("ayuda")
        assert "Comandos disponibles" in answer

    def test_unknown(self, agent):
        answer = agent.answer("clima soleado")
        assert "No entendí" in answer or "No encontré" in answer

    def test_empty_graph(self):
        g = NetworkXNormativaGraph()
        a = GraphRAGAgent(g)
        answer = a.answer("qué dice el artículo 1")
        assert "No encontré" in answer
