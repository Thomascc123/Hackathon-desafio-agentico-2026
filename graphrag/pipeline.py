#!/usr/bin/env python3
"""
Full pipeline: scrape → parse → build graph → export.

Usage:
    python -m graphrag.pipeline [--download] [--limit N]
"""

import argparse
import json
import os
import sys
import time

from .parser import (
    extract_text_from_pdf,
    parse_consolidated_reglamento,
    extract_metadata_from_resuelve,
)
from .graph_builder import NormativaGraph

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = "/tmp/normativa_pdfs"
OUTPUT_DIR = os.path.join(BASE_DIR, "graphrag_output")

# Documentos a procesar (codigo, nombre_archivo, es_consolidado)
KEY_DOCS = [
    ("31878551", "01_reglamento_pregrado_original_1981", False),
    ("35187147", "02_reglamento_pregrado_actualizado_2025", True),
    ("17917413", "03_modifica_art_130_216_2014", False),
    ("7753259",  "04_modifica_art_49_50_51_53_180_2010", False),
    ("6702815",  "05_modifica_art_63_2009", False),
    ("122606",   "06_modifica_art_215_2000", False),
]

# Categories to scrape
CATEGORIES = [
    {"label": "pregrado", "asunto": "REGLAMENTO ESTUDIANTIL DE PREGRADO"},
    {"label": "posgrado", "asunto": "REGLAMENTO ESTUDIANTIL DE POSGRADO"},
    {"label": "matricula", "keyword": "matricula"},
]


def ensure_downloaded(codigo: str, fname: str):
    """Download a document if not already present."""
    fpath = os.path.join(PDF_DIR, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        return fpath
    sys.path.insert(0, BASE_DIR)
    from download_normativa import get_document_info, download_document
    ext, img = get_document_info(codigo)
    if ext.lower() != "pdf" and ext != "-":
        print(f"  [!] {codigo}: not a PDF ({ext}), skipping")
        return None
    ok, sz = download_document(codigo, img, fpath)
    if ok:
        print(f"  Downloaded {codigo} -> {fname} ({sz // 1024} KB)")
        return fpath
    return None


def scrape_metadata(asunto=None, keyword=None, limit=None) -> list[dict]:
    """Scrape document metadata from normativa.udea.edu.co."""
    sys.path.insert(0, BASE_DIR)
    from download_normativa import fetch_page, parse_table

    all_docs = []
    page = 1
    while True:
        html = fetch_page(asunto=asunto, keyword=keyword, page=page)
        docs = parse_table(html)
        if not docs:
            break

        # Add type/authority from context
        for d in docs:
            d["tipo_documento"] = ""
            d["autoridad"] = ""
            d["asunto"] = asunto or ""
            meta = extract_metadata_from_resuelve(d.get("resuelve", ""))
            d.update(meta)

        all_docs.extend(docs)
        if limit is not None and len(all_docs) >= limit:
            all_docs = all_docs[:limit]
            break
        if len(docs) < 50:
            break
        page += 1
        time.sleep(0.3)

    return all_docs


def run_pipeline(download=False, limit=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_DIR, exist_ok=True)
    graph = NormativaGraph()

    # Step 1: Scrape metadata from website
    print("=" * 60)
    print("STEP 1: Scraping metadata from normativa.udea.edu.co")
    print("=" * 60)
    all_metadata = []
    for cat in CATEGORIES:
        print(f"\n  Category: {cat['label']}")
        docs = scrape_metadata(
            asunto=cat.get("asunto"),
            keyword=cat.get("keyword"),
            limit=limit,
        )
        print(f"  -> {len(docs)} documents found")
        for d in docs:
            d["categoria"] = cat["label"]
        all_metadata.extend(docs)

    # Step 2: Download PDFs (optional)
    if download:
        print("\n" + "=" * 60)
        print("STEP 2: Downloading key PDFs")
        print("=" * 60)
        for codigo, fname, _ in KEY_DOCS:
            fpath = ensure_downloaded(codigo, fname + ".pdf")
            if fpath:
                print(f"  OK: {fname}")

    # Step 3: Build graph from metadata
    print("\n" + "=" * 60)
    print("STEP 3: Building knowledge graph")
    print("=" * 60)
    graph.build_from_metadata(all_metadata)
    print(f"  After metadata: {graph.graph.number_of_nodes()} nodes, "
          f"{graph.graph.number_of_edges()} edges")

    # Step 4: Parse consolidated reglamento PDF
    print("\n" + "=" * 60)
    print("STEP 4: Parsing consolidated reglamento PDF")
    print("=" * 60)
    consolidated_path = os.path.join(PDF_DIR, "02_reglamento_pregrado_actualizado_2025.pdf")
    if os.path.exists(consolidated_path):
        ok, text = extract_text_from_pdf(consolidated_path)
        if ok:
            parsed = parse_consolidated_reglamento(text)
            total_arts = sum(
                sum(len(c.get("articulos", [])) for c in t.get("capitulos", [])) +
                len(t.get("articulos_directos", []))
                for t in parsed.get("titulos", [])
            )
            print(f"  Parsed: {len(parsed['titulos'])} titles, {total_arts} articles")

            # Add reglamento document node if not exists
            graph.build_from_parsed_reglamento(parsed, "31878551")
            print(f"  After parsing: {graph.graph.number_of_nodes()} nodes, "
                  f"{graph.graph.number_of_edges()} edges")
        else:
            print("  [!] Could not extract text from consolidated PDF")
    else:
        print("  [!] Consolidated PDF not found. Run with --download first.")

    # Step 5: Summary and export
    print("\n" + "=" * 60)
    print("STEP 5: Summary & Export")
    print("=" * 60)
    summary = graph.summary()
    print(f"\n  Nodes: {summary['nodes']}")
    for nt, count in summary["node_types"].items():
        if count > 0:
            print(f"    {nt.value}: {count}")
    print(f"  Edges: {summary['edges']}")
    for et, count in summary["edge_types"].items():
        if count > 0:
            print(f"    {et.value}: {count}")

    # Export JSON
    json_path = os.path.join(OUTPUT_DIR, "knowledge_graph.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph.to_json(), f, ensure_ascii=False, indent=2)
    print(f"\n  JSON export: {json_path}")

    # Export Cypher
    cypher_path = os.path.join(OUTPUT_DIR, "neo4j_import.cypher")
    with open(cypher_path, "w", encoding="utf-8") as f:
        f.write(graph.to_cypher())
    print(f"  Cypher export: {cypher_path}")

    # Print sample queries
    print("\n" + "=" * 60)
    print("SAMPLE QUERIES")
    print("=" * 60)

    # Evolution of article 63
    print("\n--- Evolución del Artículo 63 ---")
    print(graph.visualize_article_history("63"))

    # Articles about matrícula
    print("\n\n--- Artículos sobre 'matrícula' (primeros 3) ---")
    arts = graph.query_articles_by_keyword("matrícula")
    for a in arts[:3]:
        print(f"  Art. {a['articulo']}: {a['texto'][:100]}...")
        if a['modificaciones'] > 0:
            print(f"    ({a['modificaciones']} modificaciones)")

    # Document timeline for pregrado
    print("\n\n--- Línea de tiempo - Pregrado (primeros 5) ---")
    timeline = graph.query_document_timeline(asunto="REGLAMENTO ESTUDIANTIL DE PREGRADO")
    for doc in timeline[:5]:
        print(f"  {doc['fecha']} | #{doc['numero']} | {doc['resuelve'][:70]}...")

    return graph


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Pipeline for UdeA Normativa"
    )
    parser.add_argument("--download", action="store_true",
                        help="Download PDFs before building graph")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max docs per category")
    args = parser.parse_args()
    run_pipeline(download=args.download, limit=args.limit)


if __name__ == "__main__":
    main()
