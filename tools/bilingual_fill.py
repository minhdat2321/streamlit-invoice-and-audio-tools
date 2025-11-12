#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bilingual_vi_en_openxml_plus_v2.py

Bilingual document styling:
- Vietnamese text: Normal black, non-italic, Times New Roman (FIRST)
- English text: Accent1 blue, italic, Times New Roman (SECOND)
- Vietnamese always comes first, English second
- FIXED: Vietnamese style now properly applies black color
- FIXED: Improved translation coverage for English to Vietnamese
"""

import io
import os, re, math, time, copy, argparse
from typing import Optional, Dict, List, Tuple, Set
from zipfile import ZipFile
from lxml import etree

# -------- Namespaces --------
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}

# -------- OpenAI (optional) --------
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    _OPENAI_AVAILABLE = False

# -------- Language heuristics --------
_VI_DIACRITICS = (
    "ăâđêôơư"
    "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụ"
    "ýỳỷỹỵ"
    "ĂÂĐÊÔƠƯ"
    "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪỆ"
    "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤ"
    "ÝỲỶỸỴ"
)
VI_CHAR_RE = re.compile(f"[{_VI_DIACRITICS}]")

# Numeric patterns - paragraphs that are mostly numbers should not be translated
NUMERIC_RE = re.compile(r'^[\d\s\.,\-/%°#®™©∙•·]+$')
PURE_NUMBER_RE = re.compile(r'^\d+$')

ENGLISH_WORD_RE = re.compile(r'^[A-Za-z0-9\s\.,!?\-\'":;@#$%&*()+=\[\]{}<>/\\]+$')

def is_mostly_numeric(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return False
    if PURE_NUMBER_RE.match(s):
        return True
    if NUMERIC_RE.match(s):
        return True
    if len(s) <= 5 and any(c.isdigit() for c in s):
        num_count = sum(1 for c in s if c.isdigit())
        if num_count / len(s) > 0.6:
            return True
    return False

def is_vi(s: str) -> bool:
    if is_mostly_numeric(s):
        return False
    return bool(VI_CHAR_RE.search(s or ""))

def is_enish(s: str) -> bool:
    s = (s or "").strip()
    if not s or is_mostly_numeric(s):
        return False
    return not is_vi(s) and bool(ENGLISH_WORD_RE.match(s))

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace('"', '').replace("'", "")
    return s.lower()

def length_ratio(a: str, b: str) -> float:
    la = max(1, len((a or "").strip()))
    lb = max(1, len((b or "").strip()))
    return min(la, lb) / max(la, lb)

def punct_similarity(a: str, b: str) -> float:
    a_punct = re.sub(r'[A-Za-zÀ-ỹ0-9\s]', '', a or "")
    b_punct = re.sub(r'[A-Za-zÀ-ỹ0-9\s]', '', b or "")
    if not a_punct and not b_punct:
        return 1.0
    if not a_punct or not b_punct:
        return 0.0
    if a_punct in b_punct or b_punct in a_punct:
        return 0.8
    if set(a_punct) == set(b_punct):
        return 0.9
    return 0.1

def cosine(u, v):
    if not u or not v:
        return 0.0
    s = sum(a*b for a, b in zip(u, v))
    nu = math.sqrt(sum(a*a for a in u))
    nv = math.sqrt(sum(b*b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return s / (nu * nv)

# -------- XML helpers --------
def get_text_from_p(p_el) -> str:
    return "".join([(t.text or "") for t in p_el.findall(".//w:t", NS)]).strip()

def collect_paragraphs(root) -> List[dict]:
    body_like = root.find(".//w:body", NS) or root
    ps = list(body_like.findall(".//w:p", NS))
    out = []
    for i, p in enumerate(ps):
        txt = get_text_from_p(p)
        out.append({"el": p, "text": txt, "idx": i})
    return out

def first_run_rPr(src_p):
    r = src_p.find("w:r", NS)
    if r is None:
        r = src_p.find(".//w:r", NS)
    if r is not None:
        rPr = r.find("w:rPr", NS)
        if rPr is not None:
            return copy.deepcopy(rPr)
    return None

def ensure_times_new_roman(rPr):
    if rPr is None:
        return
    rFonts = rPr.find("w:rFonts", NS)
    if rFonts is None:
        rFonts = etree.SubElement(rPr, f"{{{W_NS}}}rFonts")
    rFonts.set(f"{{{W_NS}}}ascii", "Times New Roman")
    rFonts.set(f"{{{W_NS}}}hAnsi", "Times New Roman")
    rFonts.set(f"{{{W_NS}}}cs", "Times New Roman")

def get_font_size_val(rPr):
    if rPr is None:
        return None
    sz = rPr.find("w:sz", NS)
    if sz is not None and sz.get(f"{{{W_NS}}}val"):
        return sz.get(f"{{{W_NS}}}val")
    szCs = rPr.find("w:szCs", NS)
    if szCs is not None and szCs.get(f"{{{W_NS}}}val"):
        return szCs.get(f"{{{W_NS}}}val")
    return None

def set_font_size_val(rPr, val):
    if rPr is None:
        return
    sz = rPr.find("w:sz", NS)
    if sz is None:
        sz = etree.SubElement(rPr, f"{{{W_NS}}}sz")
    sz.set(f"{{{W_NS}}}val", val)
    szCs = rPr.find("w:szCs", NS)
    if szCs is None:
        szCs = etree.SubElement(rPr, f"{{{W_NS}}}szCs")
    szCs.set(f"{{{W_NS}}}val", val)

def create_vietnamese_style(rPr):
    """Vietnamese style: black, non-italic, Times New Roman"""
    if rPr is None:
        rPr = etree.Element(f"{{{W_NS}}}rPr")
    
    ensure_times_new_roman(rPr)
    
    # Remove italic if present
    i_el = rPr.find("w:i", NS)
    if i_el is not None:
        rPr.remove(i_el)
    
    # Set black color - ensure it overrides any existing color
    color = rPr.find("w:color", NS)
    if color is not None:
        rPr.remove(color)  # Remove existing color element
    color = etree.SubElement(rPr, f"{{{W_NS}}}color")
    color.set(f"{{{W_NS}}}val", "000000")  # Black
    
    return rPr

def create_english_style(rPr):
    """English style: blue, italic, Times New Roman"""
    if rPr is None:
        rPr = etree.Element(f"{{{W_NS}}}rPr")
    
    ensure_times_new_roman(rPr)
    
    # Remove italic if present, then add it
    i_el = rPr.find("w:i", NS)
    if i_el is not None:
        rPr.remove(i_el)
    etree.SubElement(rPr, f"{{{W_NS}}}i")  # Add italic
    
    # Set blue color - ensure it overrides any existing color
    color = rPr.find("w:color", NS)
    if color is not None:
        rPr.remove(color)  # Remove existing color element
    color = etree.SubElement(rPr, f"{{{W_NS}}}color")
    color.set(f"{{{W_NS}}}val", "5B9BD5")  # Accent1 blue
    
    return rPr

def enforce_language_style(p_el, is_english: bool):
    """Apply appropriate style to all runs in a paragraph based on language"""
    for r in p_el.findall(".//w:r", NS):
        rPr = r.find("w:rPr", NS)
        if rPr is None:
            rPr = etree.SubElement(r, f"{{{W_NS}}}rPr")
        if is_english:
            create_english_style(rPr)
        else:
            create_vietnamese_style(rPr)

def build_paragraph_from_source(src_p, text, is_english: bool, match_size=True):
    """Build a new paragraph with language-appropriate style"""
    p_new = etree.Element(f"{{{W_NS}}}p")
    pPr = src_p.find("w:pPr", NS)
    if pPr is not None:
        p_new.append(copy.deepcopy(pPr))

    base_rPr = first_run_rPr(src_p)
    if base_rPr is None:
        base_rPr = etree.Element(f"{{{W_NS}}}rPr")

    # Apply language-appropriate style
    if is_english:
        base_rPr = create_english_style(base_rPr)
    else:
        base_rPr = create_vietnamese_style(base_rPr)

    if match_size:
        size_val = get_font_size_val(base_rPr)
        if size_val:
            set_font_size_val(base_rPr, size_val)

    r = etree.SubElement(p_new, f"{{{W_NS}}}r")
    r.append(copy.deepcopy(base_rPr))
    t = etree.SubElement(r, f"{{{W_NS}}}t")
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return p_new

# -------- OpenAI helpers --------
class OpenAIHelper:
    def __init__(self, chat_model: str, embed_model: Optional[str], temp: float, api_key: Optional[str] = None):
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("openai==1.x not installed. pip install openai")
        key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temp = temp
        self._emb_cache: Dict[str, List[float]] = {}

    def translate_en_to_vi(self, text_en: str) -> str:
        if is_mostly_numeric(text_en):
            return text_en
            
        sys_prompt = (
            "You are a precise English-to-Vietnamese translator for technical/business documents. "
            "Return only the Vietnamese translation; no extra quotes or commentary. Preserve numbers, units, model names, punctuation. "
            "Use professional, formal Vietnamese suitable for business proposals."
        )
        user_msg = f"Translate from English to Vietnamese:\n{text_en}"
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.chat_model, temperature=self.temp,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user_msg}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"Translation attempt {attempt+1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        print(f"Failed to translate EN→VI: {text_en[:50]}...")
        return text_en  # Return original if all attempts fail

    def translate_vi_to_en(self, text_vi: str) -> str:
        if is_mostly_numeric(text_vi):
            return text_vi
            
        sys_prompt = (
            "You are a precise Vietnamese-to-English translator for technical/business documents. "
            "Return only the English translation; no extra quotes or commentary. Preserve numbers, units, model names, punctuation."
        )
        user_msg = f"Translate from Vietnamese to English:\n{text_vi}"
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.chat_model, temperature=self.temp,
                    messages=[{"role": "system", "content": sys_prompt},
                              {"role": "user", "content": user_msg}],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                print(f"Translation attempt {attempt+1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        print(f"Failed to translate VI→EN: {text_vi[:50]}...")
        return text_vi  # Return original if all attempts fail

    def embed(self, text: str):
        if not self.embed_model:
            return None
        if text in self._emb_cache:
            return self._emb_cache[text]
        for attempt in range(3):
            try:
                r = self.client.embeddings.create(model=self.embed_model, input=text)
                vec = r.data[0].embedding
                self._emb_cache[text] = vec
                return vec
            except Exception as e:
                print(f"Embedding attempt {attempt+1} failed: {e}")
                time.sleep(1.5 * (attempt + 1))
        return None

# -------- Pairing & dedup --------
def find_pairs_and_existing_norms(paras: List[dict], helper: Optional[OpenAIHelper], window: int, thresh: float, use_embeddings: bool):
    skip_vi: Set[int] = set()
    skip_en: Set[int] = set()
    existing_norms: Set[str] = set()

    n = len(paras)

    def emb(text):
        return helper.embed(text) if (use_embeddings and helper and helper.embed_model) else None

    # Skip numeric content
    skip_numeric = set()
    for i, p in enumerate(paras):
        nt = normalize_text(p["text"])
        if nt:
            existing_norms.add(nt)
        if is_mostly_numeric(p["text"]):
            skip_numeric.add(i)

    # Bi-directional pairing
    for i in range(n):
        if i in skip_vi or i in skip_en or i in skip_numeric:
            continue
        ti = (paras[i]["text"] or "").strip()
        if not ti:
            continue

        if is_vi(ti):
            e_vi = emb(ti)
            for j in range(max(0, i-window), min(n, i+window+1)):
                if j == i or j in skip_vi or j in skip_en or j in skip_numeric:
                    continue
                tj = (paras[j]["text"] or "").strip()
                if not tj or not is_enish(tj):
                    continue
                if length_ratio(ti, tj) < 0.3:
                    continue
                if punct_similarity(ti, tj) < 0.5:
                    continue
                if use_embeddings and helper and helper.embed_model:
                    e_en = emb(tj)
                    sim = cosine(e_vi, e_en) if (e_vi and e_en) else 0.0
                    if sim >= thresh:
                        skip_vi.add(i); skip_en.add(j); break
                else:
                    norm_i = normalize_text(ti)
                    norm_j = normalize_text(tj)
                    if length_ratio(norm_i, norm_j) > 0.6:
                        skip_vi.add(i); skip_en.add(j); break

        elif is_enish(ti):
            e_en = emb(ti)
            for j in range(max(0, i-window), min(n, i+window+1)):
                if j == i or j in skip_vi or j in skip_en or j in skip_numeric:
                    continue
                tj = (paras[j]["text"] or "").strip()
                if not tj or not is_vi(tj):
                    continue
                if length_ratio(ti, tj) < 0.3:
                    continue
                if punct_similarity(ti, tj) < 0.5:
                    continue
                if use_embeddings and helper and helper.embed_model:
                    e_vi = emb(tj)
                    sim = cosine(e_en, e_vi) if (e_en and e_vi) else 0.0
                    if sim >= thresh:
                        skip_en.add(i); skip_vi.add(j); break
                else:
                    norm_i = normalize_text(ti)
                    norm_j = normalize_text(tj)
                    if length_ratio(norm_i, norm_j) > 0.6:
                        skip_en.add(i); skip_vi.add(j); break

    return skip_vi, skip_en, existing_norms

def already_has_translation(neighbors: List[str], candidate: str) -> bool:
    cand_norm = normalize_text(candidate)
    for t in neighbors:
        if normalize_text(t) == cand_norm:
            return True
    return False

# -------- Part processing --------
def process_xml_part(xml_bytes: bytes, min_len: int, helper: Optional[OpenAIHelper],
                     use_embeddings: bool, embed_thresh: float, window: int) -> bytes:
    root = etree.fromstring(xml_bytes)
    paras = collect_paragraphs(root)

    # Pairing and current text inventory
    skip_vi, skip_en, existing_norms = find_pairs_and_existing_norms(
        paras, helper=helper, window=window, thresh=embed_thresh, use_embeddings=use_embeddings
    )

    # Process all paragraphs in the document
    processed_count = 0
    for info in paras:
        idx, p, txt = info["idx"], info["el"], info["text"]
        if len((txt or "").strip()) < min_len or is_mostly_numeric(txt):
            continue

        # Process English paragraphs (translate to Vietnamese)
        if is_enish(txt) and idx not in skip_en:
            # Generate Vietnamese translation
            vi_text = helper.translate_en_to_vi(txt) if helper else txt
            
            # Skip if translation is the same as original
            if normalize_text(vi_text) == normalize_text(txt):
                continue
                
            # De-dup check
            neighbors = []
            parent = p.getparent()
            pos = parent.index(p)
            for k in range(max(0, pos-window), min(len(parent), pos+window+1)):
                if k == pos: continue
                node = parent[k]
                if node.tag == f"{{{W_NS}}}p":
                    neighbors.append(get_text_from_p(node))
            
            has_similar = False
            for neighbor in neighbors:
                if normalize_text(neighbor) == normalize_text(vi_text):
                    has_similar = True
                    break
            if has_similar:
                continue
                
            # Build Vietnamese paragraph - normal black style
            vi_p = build_paragraph_from_source(p, vi_text, is_english=False, match_size=True)
            
            # Convert the original English paragraph to English style (blue italic)
            enforce_language_style(p, is_english=True)
            
            # Insert Vietnamese ABOVE the English paragraph (Vietnamese first, English second)
            parent.insert(pos, vi_p)
            processed_count += 1

        # Process Vietnamese paragraphs (translate to English)
        elif is_vi(txt) and idx not in skip_vi:
            # Generate English translation
            en_text = helper.translate_vi_to_en(txt) if helper else txt
            
            # Skip if translation is the same as original
            if normalize_text(en_text) == normalize_text(txt):
                continue
                
            # De-dup check
            neighbors = []
            parent = p.getparent()
            pos = parent.index(p)
            for k in range(max(0, pos-window), min(len(parent), pos+window+1)):
                if k == pos: continue
                node = parent[k]
                if node.tag == f"{{{W_NS}}}p":
                    neighbors.append(get_text_from_p(node))
            
            has_similar = False
            for neighbor in neighbors:
                if normalize_text(neighbor) == normalize_text(en_text):
                    has_similar = True
                    break
            if has_similar:
                continue
                
            # Build English paragraph - blue italic style
            en_p = build_paragraph_from_source(p, en_text, is_english=True, match_size=True)
            
            # Convert the original Vietnamese paragraph to Vietnamese style (normal black)
            enforce_language_style(p, is_english=False)
            
            # For Vietnamese original: Keep Vietnamese first, insert English BELOW (Vietnamese first, English second)
            # So we insert the English translation AFTER the Vietnamese paragraph
            parent.insert(pos + 1, en_p)
            processed_count += 1

    print(f"Processed {processed_count} paragraphs in this part")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")

# -------- High-level helpers --------
def process_docx_bytes(
    docx_bytes: bytes,
    *,
    headers: bool = False,
    footers: bool = False,
    min_len: int = 3,
    chat_model: Optional[str] = "gpt-4o-mini",
    embed_model: Optional[str] = None,
    use_embeddings: bool = False,
    embed_thresh: float = 0.80,
    window: int = 5,
    temp: float = 0.2,
    api_key: Optional[str] = None,
) -> bytes:
    """Process a DOCX payload and return updated bytes."""

    helper = None
    key = api_key or os.getenv("OPENAI_API_KEY")
    if _OPENAI_AVAILABLE and chat_model and key:
        helper = OpenAIHelper(
            chat_model=chat_model,
            embed_model=embed_model,
            temp=temp,
            api_key=key,
        )

    with ZipFile(io.BytesIO(docx_bytes)) as zin:
        parts = {zi.filename: zin.read(zi.filename) for zi in zin.infolist()}

    to_process = ["word/document.xml"]
    if headers:
        to_process += [name for name in parts if name.startswith("word/header") and name.endswith(".xml")]
    if footers:
        to_process += [name for name in parts if name.startswith("word/footer") and name.endswith(".xml")]

    for part in to_process:
        if part in parts:
            parts[part] = process_xml_part(
                parts[part],
                min_len=min_len,
                helper=helper,
                use_embeddings=use_embeddings,
                embed_thresh=embed_thresh,
                window=window,
            )

    out_buf = io.BytesIO()
    with ZipFile(out_buf, "w") as zout:
        for name, data in parts.items():
            zout.writestr(name, data)

    return out_buf.getvalue()


def process_docx(
    input_path: str,
    output_path: str,
    **kwargs,
):
    with open(input_path, "rb") as fin:
        result = process_docx_bytes(fin.read(), **kwargs)
    with open(output_path, "wb") as fout:
        fout.write(result)


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Create bilingual document with Vietnamese first, English second.")
    ap.add_argument("input_docx")
    ap.add_argument("output_docx")
    ap.add_argument("--headers", action="store_true", help="Process headers.")
    ap.add_argument("--footers", action="store_true", help="Process footers.")
    ap.add_argument("--min-len", type=int, default=3, help="Minimum paragraph length to consider (default 3).")
    ap.add_argument("--chat-model", default="gpt-4o-mini", help="OpenAI chat model.")
    ap.add_argument("--embed-model", default=None, help="OpenAI embeddings model, e.g. text-embedding-3-large.")
    ap.add_argument("--use-embeddings", action="store_true", help="Use embeddings to confirm VN–EN pairs.")
    ap.add_argument("--embed-thresh", type=float, default=0.80, help="Cosine threshold for pairing (default 0.80).")
    ap.add_argument("--window", type=int, default=5, help="Neighborhood window for pairing and dedup (default 5).")
    ap.add_argument("--temp", type=float, default=0.2, help="Chat translation temperature.")
    args = ap.parse_args()

    process_docx(
        args.input_docx,
        args.output_docx,
        headers=args.headers,
        footers=args.footers,
        min_len=args.min_len,
        chat_model=args.chat_model,
        embed_model=args.embed_model,
        use_embeddings=args.use_embeddings,
        embed_thresh=args.embed_thresh,
        window=args.window,
        temp=args.temp,
    )

    print(f"Done. Wrote: {args.output_docx}")


if __name__ == "__main__":
    main()
