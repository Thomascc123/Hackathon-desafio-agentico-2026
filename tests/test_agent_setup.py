import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.models import (
    ArticleTextResult,
    KeywordSearchResult,
    EvolutionItem,
    DocumentSearchResult,
)
from agent.agent_setup import (
    _format_data_for_llm,
    _extract_sources_from_text,
    _convert_history,
)


# ── _format_data_for_llm ───────────────────────────────────────────

class TestFormatDataForLLM:
    def test_empty_data(self):
        assert _format_data_for_llm([], "article_text") == "No se encontraron resultados."
        assert _format_data_for_llm(None, "article_text") == "No se encontraron resultados."

    def test_article_text(self):
        data = [
            ArticleTextResult(numero="63", texto="Texto del artículo 63", documento="Reglamento"),
        ]
        result = _format_data_for_llm(data, "article_text")
        assert "Artículo 63" in result
        assert "Texto del artículo 63" in result
        assert "Documento: Reglamento" in result

    def test_keyword_search(self):
        data = [
            KeywordSearchResult(articulo="10", texto="Texto del artículo 10",
                                documento_asunto="MATRICULA"),
        ]
        result = _format_data_for_llm(data, "keyword_search")
        assert "Art. 10" in result
        assert "(MATRICULA)" in result
        assert "Texto del artículo 10" in result

    def test_evolution(self):
        data = [
            EvolutionItem(modificado_por="AS 458/2019", anio="2019",
                          accion="Modifica el artículo 63"),
            EvolutionItem(articulo="63", texto_actual="Texto vigente", num_modificaciones=1),
        ]
        result = _format_data_for_llm(data, "evolution")
        assert "AS 458/2019" in result
        assert "2019" in result

    def test_document_search(self):
        data = [
            DocumentSearchResult(numero="623", fecha="2025-06-15",
                                 resuelve="Modificar el reglamento"),
        ]
        result = _format_data_for_llm(data, "document_search")
        assert "#623" in result
        assert "2025-06-15" in result
        assert "Modificar el reglamento" in result

    def test_mixed_typed_and_raw_dicts(self):
        """Should handle both Pydantic models and raw dicts."""
        data = [
            {"numero": "1", "texto": "Dict article", "modificaciones": 0},
            ArticleTextResult(numero="2", texto="Model article"),
        ]
        result = _format_data_for_llm(data, "article_text")
        assert "Artículo 1" in result
        assert "Dict article" in result
        assert "Model article" in result
        assert "Model article" in result


# ── _extract_sources_from_text ─────────────────────────────────────

class TestExtractSources:
    def test_extract_articles(self):
        text = "Según el Artículo 63 del reglamento... Ver también Art. 10."
        sources = _extract_sources_from_text(text)
        assert "Art. 63" in sources
        assert "Art. 10" in sources

    def test_extract_documents(self):
        text = "Modificado por AS 458/2019 y AS 623/2025."
        sources = _extract_sources_from_text(text)
        assert "AS 458/2019" in sources
        assert "AS 623/2025" in sources

    def test_extract_no_sources(self):
        text = "No hay referencias en este texto."
        sources = _extract_sources_from_text(text)
        assert sources == []

    def test_extract_mixed(self):
        text = "Art. 63 modificado por AS 458/2019."
        sources = _extract_sources_from_text(text)
        assert "Art. 63" in sources
        assert "AS 458/2019" in sources
        assert len(sources) == 2

    def test_deduplication(self):
        text = "Art. 63. Ver también Art. 63."
        sources = _extract_sources_from_text(text)
        assert len(sources) == 1


# ── _convert_history ───────────────────────────────────────────────

class TestConvertHistory:
    def test_empty_history(self):
        result = _convert_history([])
        assert result == []

    def test_user_and_assistant(self):
        history = [
            {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "Adiós"},
        ]
        result = _convert_history(history)
        assert len(result) == 2

    def test_skips_empty_content(self):
        history = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "Hola"},
        ]
        result = _convert_history(history)
        assert len(result) == 1
