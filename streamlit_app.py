import io
import os
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile

import streamlit as st
import pandas as pd

from utils.openai_client import get_openai_client
from tools.recording_text_extract import transcribe_file_like
from tools.invoices_extract import extract_records_from_images

# NEW: bilingual module
from tools.bilingual_fill import process_docx as process_bilingual

APP_TITLE = "AI Tools: Transcribe • Invoice Extract • Bilingual DOCX"

st.set_page_config(page_title=APP_TITLE, page_icon="🎛️", layout="wide")
st.title(APP_TITLE)
st.caption("Built with Streamlit • OpenAI API • Pydantic • Pandas • python-docx")

# --- API Key handling ---
def resolve_api_key():
    # Priority: Streamlit secrets → env OPENAI_API_KEY → manual input
    key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    return key

openai_key = resolve_api_key()
manual_key = None
if not openai_key:
    with st.sidebar:
        st.info("No OpenAI key detected in secrets/env. Enter a temporary key below.")
        manual_key = st.text_input("OPENAI_API_KEY", type="password")
openai_key = openai_key or manual_key

if not openai_key:
    st.warning("Please set OPENAI_API_KEY in .streamlit/secrets.toml, environment, or enter it in the sidebar.")

# Model pickers (keep conservative defaults)
with st.sidebar:
    st.header("Settings")
    st.write("**General**")
    st.text_input("Organization (optional)", key="org")

    st.write("**Transcription**")
    st.session_state.setdefault("asr_model", "gpt-4o-transcribe")
    asr_model = st.text_input("ASR model", value=st.session_state["asr_model"], help="e.g., gpt-4o-transcribe or whisper-1")
    asr_lang = st.text_input("Language hint", value="vi", help="ISO code hint, e.g., vi, en")
    asr_fmt = st.selectbox("Output format", ["txt", "srt", "vtt"], index=0)

    st.write("**Invoice Extract**")
    st.session_state.setdefault("invoice_model", "gpt-4o")
    invoice_model = st.text_input("Vision model", value=st.session_state["invoice_model"], help="e.g., gpt-4o")

    st.write("**Bilingual DOCX**")
    st.session_state.setdefault("bilingual_model", "gpt-4o-mini")
    bilingual_model = st.text_input("Translation model", value=st.session_state["bilingual_model"], help="e.g., gpt-4o-mini")
    bilingual_labels = st.checkbox("Show [VI]/[EN] labels", value=True)
    table_mode = st.radio("Table mode", ["stack", "inline"], index=0, help="How to place EN in table cells")

client = get_openai_client(openai_key, org=st.session_state.get("org")) if openai_key else None

st.divider()

# ==================================
# Tabs
# ==================================

t1, t2, t3 = st.tabs(["🔊 Transcribe Audio", "🧾 Extract Invoices", "📝 Bilingual DOCX (VI → EN)"])

# ==================================
# Tab 1 — Audio → Text transcription
# ==================================
with t1:
    st.subheader("Upload audio to transcribe")
    audio = st.file_uploader("Select an audio file (mp3, m4a, wav, aac, flac, ogg)", type=["mp3", "m4a", "wav", "aac", "flac", "ogg"])
    colA, colB = st.columns([1,1])

    with colA:
        if audio:
            st.audio(audio)
    with colB:
        st.write("\n")
        st.caption("Choose format (TXT by default; SRT/VTT include timestamps)")
        run_asr = st.button("Transcribe now", use_container_width=True, disabled=not (client and audio))

    if run_asr and client and audio:
        with st.spinner("Transcribing…"):
            text = transcribe_file_like(
                client=client,
                file_like=io.BytesIO(audio.getvalue()),
                filename=audio.name,
                model=asr_model,
                language=asr_lang,
                response_format=None if asr_fmt == "txt" else asr_fmt,
            )
        if text:
            st.success("Done!")
            st.code(text[:2000] + ("…" if len(text) > 2000 else ""), language="text")
            suffix = f".{asr_fmt if asr_fmt in ('srt','vtt') else 'txt'}"
            out_name = Path(audio.name).with_suffix(suffix).name
            st.download_button("Download result", data=text.encode("utf-8"), file_name=out_name, mime="text/plain")
        else:
            st.error("No text produced.")

# ==================================
# Tab 2 — Invoice images → table
# ==================================
with t2:
    st.subheader("Upload invoice images (PNG/JPG/WebP/BMP/TIFF)")
    imgs = st.file_uploader("Select one or multiple images", type=["png","jpg","jpeg","webp","bmp","tiff"], accept_multiple_files=True)
    run_ie = st.button("Extract to table", use_container_width=True, disabled=not (client and imgs))

    if run_ie and client and imgs:
        with st.spinner("Extracting… this may take a moment for multiple files"):
            files = [(img.name, img.getvalue()) for img in imgs]
            df, notes = extract_records_from_images(client=client, files=files, model=invoice_model)

        st.success("Extraction complete")
        st.dataframe(df, use_container_width=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as wr:
            df.to_excel(wr, index=False, sheet_name="Invoices")
        st.download_button("Download CSV", data=csv_bytes, file_name=f"invoices_{ts}.csv", mime="text/csv")
        st.download_button("Download XLSX", data=xlsx_buf.getvalue(), file_name=f"invoices_{ts}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if notes:
            with st.expander("Confidence notes / warnings"):
                for n in notes:
                    st.markdown(f"- {n}")

# ==================================
# Tab 3 — Bilingual DOCX
# ==================================
with t3:
    st.subheader("Upload a Word document (.docx) to make it bilingual")
    st.caption("Vietnamese will be kept first; English will be appended and styled as *italic* + Accent 1 (blue). Bold/italic spans are preserved exactly.")

    docx_file = st.file_uploader("Select a .docx file", type=["docx"], key="docx_upload")
    run_bi = st.button("Convert to Bilingual DOCX", use_container_width=True, disabled=not (client and docx_file))

    # Translator wrapper that matches the tags contract in tools/bilingual_fill.py
    def make_openai_translator(_client, _model: str):
        # Expecting tools.bilingual_fill to call translate_fn(tagged_text, src_lang)
        def translate_fn(text_with_tags: str, src_lang: str) -> str:
            target = "English" if src_lang == "vi" else "Vietnamese"
            sys = (
                "You are a professional translator. Translate the content while "
                "STRICTLY preserving these inline tags: <b>, </b>, <i>, </i>, <bi>, </bi>. "
                "Do not remove, add, or reorder tags; keep them around the same phrases they wrap. "
                "Preserve numbers, units, and proper nouns."
            )
            resp = _client.responses.create(
                model=_model,
                input=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"Translate to {target}:\n\n{text_with_tags}"}
                ]
            )
            return resp.output_text or ""
        return translate_fn

    if run_bi and client and docx_file:
        translate_fn = make_openai_translator(client, bilingual_model)

        # Write input to a temp file, process, and serve
        with NamedTemporaryFile(delete=False, suffix=".docx") as in_tmp:
            in_tmp.write(docx_file.getvalue())
            in_tmp.flush()
            in_path = in_tmp.name

        out_tmp = NamedTemporaryFile(delete=False, suffix=".docx")
        out_tmp.close()

        with st.spinner("Translating and formatting…"):
            # The bilingual module already enforces:
            # - VI first, EN second
            # - EN italic + Accent1 blue
            # - Preserve bold/italic spans
            process_bilingual(
                in_path=in_path,
                out_path=out_tmp.name,
                translate_fn=translate_fn,
                add_labels=bilingual_labels,
                table_mode=table_mode,
            )

        with open(out_tmp.name, "rb") as f:
            st.download_button(
                label="Download bilingual .docx",
                data=f.read(),
                file_name=Path(docx_file.name).with_suffix(".bilingual.docx").name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        st.success("Done! Vietnamese first, English italic + Accent 1.")
