import io
import os
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

from utils.openai_client import get_openai_client
from tools.recording_text_extract import transcribe_file_like
from tools.invoices_extract import extract_records_from_images

APP_TITLE = "AI Tools: Transcribe & Invoice Extract"

st.set_page_config(page_title=APP_TITLE, page_icon="🎛️", layout="wide")
st.title(APP_TITLE)
st.caption("Built with Streamlit • OpenAI API • Pydantic • Pandas")

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

client = get_openai_client(openai_key, org=st.session_state.get("org")) if openai_key else None

st.divider()

# ==================================
# Tab 1 — Audio → Text transcription
# ==================================

t1, t2 = st.tabs(["🔊 Transcribe Audio", "🧾 Extract Invoices"])

with t1:
    st.subheader("Upload audio to transcribe")
    audio = st.file_uploader("Select an audio file (mp3, m4a, wav)", type=["mp3", "m4a", "wav", "aac", "flac", "ogg"])
    colA, colB = st.columns([1,1])

    with colA:
        st.audio(audio) if audio else None
    with colB:
        st.write("\n")
        st.caption("Choose format (TXT by default; SRT/VTT include timestamps)")
        run_asr = st.button("Transcribe now", use_container_width=True, disabled=not (client and audio))

    if run_asr and client and audio:
        with st.spinner("Transcribing…"):
            # Use file-like interface and chosen options
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
            # Convert uploaded images into in-memory bytes list with names
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