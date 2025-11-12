import io
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from utils.openai_client import get_openai_client
from tools.audio_to_conversation import transcribe_meeting
from tools.invoices_extract import extract_records_from_images
from tools.bilingual_fill import process_docx_bytes

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

# Sidebar settings
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
    st.text_input("Default translation model", key="bilingual_model", help="e.g., gpt-4o-mini")

client = get_openai_client(openai_key, org=st.session_state.get("org")) if openai_key else None

st.divider()

# ==============================
# Tabs
# ==============================
t1, t2, t3 = st.tabs(["🔊 Transcribe Audio", "🧾 Extract Invoices", "📝 Bilingual DOCX (VI → EN)"])

# ==============================
# Tab 1 — Audio → Text
# ==============================
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
            text = transcribe_meeting(
                client=client,
                file_like=io.BytesIO(audio.getvalue()),
                filename=audio.name,
                model_asr=asr_model,
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

# ==============================
# Tab 2 — Invoice images → table
# ==============================
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

# ==============================
# Tab 3 — Bilingual DOCX
# ==============================
with t3:
    st.subheader("Upload a Word document (.docx) to make it bilingual")
    st.caption(
        "Vietnamese stays first; English is appended and styled as *italic* + Accent 1 (blue)."
        " Paragraph heuristics are shared with the bilingual_fill.py CLI."
    )

    docx_file = st.file_uploader("Select a .docx file", type=["docx"], key="docx_upload")

    col_left, col_right = st.columns(2)
    with col_left:
        process_headers = st.checkbox("Process headers", value=False)
        process_footers = st.checkbox("Process footers", value=False)
        min_len = st.number_input(
            "Minimum paragraph length",
            min_value=0,
            max_value=200,
            value=3,
            step=1,
            help="Paragraphs shorter than this are skipped.",
        )
        window = st.slider(
            "Neighbor window for pairing/dedup",
            min_value=1,
            max_value=20,
            value=5,
            help="How many nearby paragraphs to inspect when pairing and avoiding duplicates.",
        )

    with col_right:
        chat_model = st.text_input(
            "Chat translation model",
            value=st.session_state.get("bilingual_model", "gpt-4o-mini"),
            key="bilingual_chat_model",
            help="Leave blank to skip OpenAI translation (styles only).",
        )
        temp = st.slider(
            "Translation temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )
        use_embeddings = st.checkbox(
            "Use embeddings for VN–EN pairing",
            value=False,
            help="Requires an embeddings model and OpenAI key.",
        )
        if use_embeddings:
            embed_model = st.text_input(
                "Embedding model",
                value=st.session_state.get("bilingual_embed_model", ""),
                key="bilingual_embed_model",
                help="e.g., text-embedding-3-large",
            )
            embed_thresh = st.slider(
                "Embedding cosine threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.80,
                step=0.01,
                help="Higher = stricter pairing when embeddings are enabled.",
            )
        else:
            embed_model = ""
            embed_thresh = 0.80
            st.caption("Embeddings disabled → using heuristic length/punctuation pairing only.")

    run_bi = st.button(
        "Convert to bilingual DOCX",
        use_container_width=True,
        disabled=docx_file is None,
    )

    if run_bi and docx_file:
        if use_embeddings and not embed_model.strip():
            st.error("Please provide an embeddings model or disable embeddings.")
        else:
            chat_model_opt = chat_model.strip() or None
            embed_model_opt = embed_model.strip() or None

            with st.spinner("Processing document with bilingual_fill.py…"):
                result_bytes = process_docx_bytes(
                    docx_file.getvalue(),
                    headers=process_headers,
                    footers=process_footers,
                    min_len=int(min_len),
                    chat_model=chat_model_opt,
                    embed_model=embed_model_opt,
                    use_embeddings=use_embeddings,
                    embed_thresh=float(embed_thresh),
                    window=int(window),
                    temp=float(temp),
                    api_key=openai_key,
                )

            st.download_button(
                label="Download bilingual .docx",
                data=result_bytes,
                file_name=Path(docx_file.name).with_suffix(".bilingual.docx").name,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            st.success("Done!")
