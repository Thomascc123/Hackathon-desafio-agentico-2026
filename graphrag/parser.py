"""
PDF parser for UdeA normative documents.
Extracts articles, chapters, titles, and modification annotations
from the consolidated reglamento estudiantil PDFs using position-based parsing.
"""

import re
import logging
import fitz

logger = logging.getLogger(__name__)

_HAS_OCR = False
try:
    import pytesseract
    _HAS_OCR = True
except ImportError:
    pass

ROMAN_WORDS = r'PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|SÉPTIMO|OCTAVO|NOVENO|DÉCIMO|UNDÉCIMO|DUODÉCIMO'

PAT_TITULO = re.compile(r'TÍTULO\s+(' + ROMAN_WORDS + r')', re.IGNORECASE)
PAT_CAPITULO = re.compile(r'CAPÍTULO\s+([IVXLCDM]+)', re.IGNORECASE)
PAT_ARTICULO = re.compile(r'ARTÍCULO\s+(\d+)\s*[\.\–\-]?\s*', re.IGNORECASE)
PAT_PARAGRAFO = re.compile(
    r'PARÁGRAFO\s*(\d+)?\s*[\.\–\-]?\s*(.*?)(?=PARÁGRAFO\s*\d*|ARTÍCULO\s+\d+|CAPÍTULO|TÍTULO|$)',
    re.DOTALL | re.IGNORECASE,
)
PAT_MODIFICACION = re.compile(
    r'\((?:Modificado|Adicionado|Modificados|Sustituido|Derogado)\s+por\s+(?:el|la)\s+'
    r'(AS|AA|RS|RA|RR|CR|DR|Acuerdo\s+Superior|Resolución\s+Rectoral|Acuerdo\s+Académico)'
    r'\s*[\.\–\-]?\s*(\d+(?:[A-Za-z])?)\s*(?:/\s*(\d{4}))?'
    r'(?:,\s*(?:artículo|arts\.?)\s*[\.\–\-]?\s*(\d+(?:\s*(?:,|y)\s*\d+)*))?\)',
    re.IGNORECASE,
)


def extract_text_from_pdf(pdf_path: str, use_ocr: bool = True) -> tuple[bool, str]:
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()

        if text.strip():
            # Text extraction succeeded — check quality
            chars_per_page = len(text.strip()) / max(num_pages, 1)
            if chars_per_page > 50:
                return True, text
            logger.info("Low text density (%.1f chars/page), attempting OCR", chars_per_page)

        # Fallback to OCR if text is empty or too sparse
        if use_ocr and _HAS_OCR:
            logger.info("Running OCR on %s (%d pages)", pdf_path, num_pages)
            ocr_text = _ocr_pdf(pdf_path)
            if ocr_text.strip():
                return True, ocr_text
            return False, ""

        if text.strip():
            return True, text
        return False, ""

    except Exception as e:
        logger.warning("Error extracting text from %s: %s", pdf_path, e)
        return False, str(e)


def _ocr_pdf(pdf_path: str) -> str:
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR quality
        zoom = 300 / 72
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img, lang="spa")
        pages_text.append(text)
    doc.close()
    return "\n".join(pages_text)


def _find_elements(body: str) -> list[dict]:
    elements = []
    for m in PAT_TITULO.finditer(body):
        elements.append({"type": "titulo", "id": m.group(1), "start": m.start(), "end": m.end()})
    for m in PAT_CAPITULO.finditer(body):
        elements.append({"type": "capitulo", "id": m.group(1), "start": m.start(), "end": m.end()})
    for m in PAT_ARTICULO.finditer(body):
        num = int(m.group(1))
        if 1 <= num <= 300:
            elements.append({"type": "articulo", "id": m.group(1), "num": num, "start": m.start(), "end": m.end()})
    elements.sort(key=lambda x: x["start"])
    return elements


def _get_body_text(body: str, elem: dict, next_elem: dict | None) -> str:
    """Get text content of an element (from its end to the start of next sibling)."""
    start = elem["end"]
    end = next_elem["start"] if next_elem else len(body)
    return body[start:end].strip()


def _parse_article_text(text: str) -> dict:
    paras = []
    for pm in PAT_PARAGRAFO.finditer(text):
        paras.append({
            "numero": pm.group(1) or "1",
            "texto": pm.group(2).strip(),
        })
    mods = []
    for m in PAT_MODIFICACION.finditer(text):
        mods.append({
            "tipo_norma": m.group(1).strip(),
            "numero": m.group(2).strip(),
            "anio": m.group(3) or "",
            "articulos_ref": m.group(4) or "",
        })
    return {"paragrafos": paras, "modificaciones": mods}


def parse_consolidated_reglamento(text: str) -> dict:
    acuerdo_idx = re.search(r'\bACUERDA\b', text, re.IGNORECASE)
    body_start = acuerdo_idx.end() if acuerdo_idx else 0
    body = text[body_start:]

    elements = _find_elements(body)
    result = {"titulos": []}

    # Build hierarchy from flat position-ordered elements
    titulo_stack: list[dict] = []
    capitulo_stack: list[dict] = []

    for i, elem in enumerate(elements):
        next_elem = elements[i + 1] if i + 1 < len(elements) else None
        etype = elem["type"]

        if etype == "titulo":
            t = {"numero": elem["id"], "capitulos": [], "articulos_directos": []}
            titulo_stack = [t]
            capitulo_stack = []
            result["titulos"].append(t)

        elif etype == "capitulo":
            if not titulo_stack:
                t = {"numero": "SIN_TITULO", "capitulos": [], "articulos_directos": []}
                titulo_stack = [t]
                result["titulos"].append(t)
            c = {"numero": elem["id"], "articulos": []}
            capitulo_stack = [c]
            titulo_stack[-1]["capitulos"].append(c)

        elif etype == "articulo":
            art_text = _get_body_text(body, elem, next_elem)
            extra = _parse_article_text(art_text)
            a = {
                "numero": elem["id"],
                "texto": re.sub(r'\s+', ' ', art_text).strip(),
                **extra,
            }
            if capitulo_stack:
                capitulo_stack[-1]["articulos"].append(a)
            elif titulo_stack:
                titulo_stack[-1]["articulos_directos"].append(a)
            else:
                pass  # orphan article

    return result


def extract_modificaciones_direct(text: str) -> list[dict]:
    mods = []
    for m in PAT_MODIFICACION.finditer(text):
        mods.append({
            "tipo_norma": m.group(1).strip(),
            "numero": m.group(2).strip(),
            "anio": m.group(3) or "",
            "articulos_ref": m.group(4) or "",
        })
    return mods


def extract_metadata_from_resuelve(resuelve: str) -> dict:
    info = {
        "accion": "",
        "articulos_modificados": [],
        "documento_modificado": "",
    }
    action_patterns = [
        (r'MODIFICAR|MODIFICA|MODIFIQUE', 'modifica'),
        (r'DEROGAR|DEROGA|DEROGUE', 'deroga'),
        (r'ADICIONAR|ADICIONA|ADICIONE', 'adiciona'),
        (r'SUPRIMIR|SUPRIME|SUPRIMA', 'suprime'),
        (r'SUSTITUIR|SUSTITUYE|SUSTITUYA', 'sustituye'),
        (r'EXPIDE|EXPEDIR', 'expide'),
        (r'REGLAMENTAR|REGLAMENTA', 'reglamenta'),
        (r'FIJAR', 'fija'),
    ]
    for pattern, action in action_patterns:
        if re.search(pattern, resuelve, re.IGNORECASE):
            info["accion"] = action
            break
    nums = re.findall(r'(?:artículos?|arts\.?)\s*(\d+(?:\s*(?:,|y)\s*\d+)*)', resuelve, re.IGNORECASE)
    for m in nums:
        info["articulos_modificados"].extend(int(n) for n in re.findall(r'\d+', m))
    nums2 = re.findall(r'ARTICULO\s+(\d+)', resuelve, re.IGNORECASE)
    info["articulos_modificados"].extend(int(n) for n in nums2)
    info["articulos_modificados"] = sorted(set(info["articulos_modificados"]))
    doc_refs = [
        r'ACUERDO\s+SUPERIOR\s+(\d+(?:[A-Za-z])?)\s+DE\s+(\d{4})',
        r'ACUERDO\s+(\d+(?:[A-Za-z])?)\s+DE\s+(\d{4})',
    ]
    for pat in doc_refs:
        m = re.search(pat, resuelve, re.IGNORECASE)
        if m:
            info["documento_modificado"] = f"Acuerdo Superior {m.group(1)} de {m.group(2)}"
            break
    return info


if __name__ == "__main__":
    path = "/tmp/normativa_pdfs/02_reglamento_pregrado_actualizado_2025.pdf"
    ok, text = extract_text_from_pdf(path)
    if not ok:
        print("No text extracted")
        exit(1)

    result = parse_consolidated_reglamento(text)
    titles = result["titulos"]

    total_arts = 0
    for t in titles:
        chaps = t.get("capitulos", [])
        dirs = t.get("articulos_directos", [])
        ct = sum(len(c["articulos"]) for c in chaps) + len(dirs)
        total_arts += ct
        print(f"TÍTULO {t['numero']}: {len(chaps)} capítulos, {len(dirs)} artículos directos, {ct} total")

    print(f"\nTotal: {len(titles)} títulos, {total_arts} artículos")

    # Show a sample article with modifications
    for t in titles:
        for c in t.get("capitulos", []):
            for a in c["articulos"][:5]:
                if a.get("modificaciones"):
                    print(f"\n  Art. {a['numero']}: {a['texto'][:80]}...")
                    for m in a["modificaciones"]:
                        print(f"    -> Modificado por {m['tipo_norma']} {m['numero']}/{m['anio']}")
