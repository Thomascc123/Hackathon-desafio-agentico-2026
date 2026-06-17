#!/usr/bin/env python3
"""
Download normative documents from normativa.udea.edu.co for:
  - Reglamento Estudiantil de Pregrado
  - Reglamento Estudiantil de Posgrado
  - Documents related to Matrícula (keyword search)

Usage:
    python download_normativa.py [--output-dir ./normativa_downloads]
"""

import argparse
import os
import re
import sys
import time
import requests
from urllib.parse import urljoin

BASE_URL = "https://normativa.udea.edu.co"
CONSULTAR_URL = urljoin(BASE_URL, "/Documentos/Consultar")
EXTENSION_URL = urljoin(BASE_URL, "/Documentos/ExtensionDocumento")
DOWNLOAD_URL = urljoin(BASE_URL, "/Documentos/Documento")

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})

REQUEST_DELAY = 0.5  # seconds between requests to be polite


def fetch_page(asunto=None, keyword=None, page=1, ordenarpor="indice2 DESC"):
    """Search the normativa site and return parsed table rows."""
    data = {
        "tipobusqueda": "indices",
        "restringido": "no",
        "ordenarpor": ordenarpor,
        "CurrentPage": str(page),
        "tipodocumento": "",
        "dependencia": "",
        "asunto": asunto or "",
        "fecha": "",
        "buscartodo": keyword or "",
    }
    resp = SESSION.post(CONSULTAR_URL, data=data, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_table(html):
    """Extract document rows from HTML table."""
    rows = re.findall(r'<tr class="documento">(.*?)</tr>', html, re.DOTALL)
    docs = []
    for r in rows:
        tds = re.findall(r'<td>(.*?)</td>', r, re.DOTALL)
        if len(tds) < 5:
            continue
        m = re.search(r"verdocumento\('(\d+)'", tds[0])
        if not m:
            continue
        codigo = m.group(1)
        numero = re.sub(r'<[^>]+>', '', tds[0]).strip()
        fecha = re.sub(r'<[^>]+>', '', tds[1]).strip()
        vigencia = re.sub(r'<[^>]+>', '', tds[2]).strip()
        medio = re.sub(r'<[^>]+>', '', tds[3]).strip()
        resuelve = re.sub(r'<[^>]+>', '', tds[4]).strip()
        # Clean up HTML entities
        resuelve = resuelve.replace('&#211;', 'Ó').replace('&#205;', 'Í')
        resuelve = resuelve.replace('&#201;', 'É').replace('&#193;', 'Á')
        resuelve = resuelve.replace('&#209;', 'Ñ').replace('&#218;', 'Ú')
        resuelve = resuelve.replace('\u00cd', 'Í').replace('\u00d3', 'Ó')
        resuelve = resuelve.replace('\u00c1', 'Á').replace('\u00c9', 'É')
        resuelve = resuelve.replace('\u00da', 'Ú').replace('\u00d1', 'Ñ')
        resuelve = re.sub(r'\s+', ' ', resuelve).strip()

        normas_rel = re.sub(r'<[^>]+>', '', tds[5]).strip() if len(tds) > 5 else ""

        docs.append({
            "codigo": codigo,
            "numero": numero,
            "fecha": fecha,
            "vigencia": vigencia,
            "medio": medio,
            "resuelve": resuelve,
            "normas_relacionadas": normas_rel,
        })
    return docs


def get_total_pages(html):
    """Extract total page count from pager div."""
    # Look for page numbers in the pager
    pages = re.findall(r'<span class="current">(\d+)</span>', html)
    if pages:
        return int(pages[0])
    # Check if there's only one page
    if re.search(r'<tr class="documento">', html):
        # If no pagination controls beyond simple prev/next, it's likely 1 page
        return 1
    return 1


def get_document_info(codigo):
    """Check document extension and get codigoimagen via AJAX endpoint."""
    try:
        resp = SESSION.post(EXTENSION_URL, data={"codigodocumento": codigo}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("extension", "pdf"), data.get("codigoimagen", "")
    except Exception as e:
        print(f"    [!] Error checking extension for {codigo}: {e}")
        return "pdf", ""


def download_document(codigo, codigoimagen, output_path):
    """Download a document by its code and image code."""
    url = f"{DOWNLOAD_URL}?codigodocumento={codigo}&codigoimagen={codigoimagen}"
    try:
        resp = SESSION.get(url, timeout=60)
        resp.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True, len(resp.content)
    except Exception as e:
        print(f"    [!] Download failed for {codigo}: {e}")
        return False, 0


def sanitize_filename(name, max_len=100):
    """Turn a string into a safe filename."""
    safe = re.sub(r'[^\w\s\-]', '', name)
    safe = re.sub(r'\s+', '_', safe.strip())
    return safe[:max_len]


def download_category(label, asunto=None, keyword=None, output_dir="downloads", limit=None):
    """Download all documents for a given category."""
    cat_dir = os.path.join(output_dir, sanitize_filename(label.lower().replace(" ", "_")))
    os.makedirs(cat_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CATEGORY: {label}")
    print(f"{'='*70}")

    page = 1
    total_downloaded = 0
    total_skipped = 0
    while True:
        print(f"  Fetching page {page}...")
        html = fetch_page(asunto=asunto, keyword=keyword, page=page)
        docs = parse_table(html)

        if not docs:
            print(f"  No more documents found. Stopping.")
            break

        limit_reached = False
        for doc in docs:
            if limit is not None and total_downloaded >= limit:
                limit_reached = True
                break

            codigo = doc["codigo"]
            fecha_short = doc["fecha"].replace("/", "")
            resuelve_short = sanitize_filename(doc["resuelve"], 80) or f"doc_{codigo}"
            fname = f"{fecha_short}_{doc['numero']}_{resuelve_short}.pdf"
            fpath = os.path.join(cat_dir, fname)

            # Check if already downloaded
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                total_skipped += 1
                continue

            print(f"    [{codigo}] {doc['numero']} - {doc['fecha']} - {doc['resuelve'][:80]}...")

            ext, codigoimagen = get_document_info(codigo)
            if ext.lower() != "pdf" and ext != "-":
                print(f"      Extension is '{ext}', skipping (not PDF).")
                total_skipped += 1
                continue

            success, size = download_document(codigo, codigoimagen or "", fpath)
            if success:
                print(f"      Downloaded: {size/1024:.1f} KB -> {fname}")
                total_downloaded += 1
            else:
                total_skipped += 1

            time.sleep(REQUEST_DELAY)

        if limit_reached:
            break

        # Check if there are more pages
        if len(docs) < 50:  # less than a full page means it's the last page
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    print(f"  Done: {total_downloaded} downloaded, {total_skipped} skipped.")
    return total_downloaded, total_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Download normative documents from normativa.udea.edu.co"
    )
    parser.add_argument(
        "--output-dir",
        default="./normativa_downloads",
        help="Output directory (default: ./normativa_downloads)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max documents to download per category (default: all)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    limit = args.limit
    os.makedirs(output_dir, exist_ok=True)

    categories = [
        {
            "label": "Reglamento Estudiantil de Pregrado",
            "asunto": "REGLAMENTO ESTUDIANTIL DE PREGRADO",
        },
        {
            "label": "Reglamento Estudiantil de Posgrado",
            "asunto": "REGLAMENTO ESTUDIANTIL DE POSGRADO",
        },
        {
            "label": "Matricula (keyword search)",
            "keyword": "matricula",
        },
    ]

    total_dl = 0
    total_sk = 0
    for cat in categories:
        dl, sk = download_category(
            label=cat["label"],
            asunto=cat.get("asunto"),
            keyword=cat.get("keyword"),
            output_dir=output_dir,
            limit=limit,
        )
        total_dl += dl
        total_sk += sk

    print(f"\n{'='*70}")
    print(f"COMPLETE: {total_dl} documents downloaded, {total_sk} skipped.")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
