"""
streamlit_app.py
Streamlit wrapper for bilingual_fill.py
- Upload .docx, choose options, provide OpenAI API key (or set env var),
  then download bilingual .docx.
"""

import io
import os
import streamlit as st
from tempfile import NamedTemporaryFile

from tools.bilingual_fill import process_docx
from tools.bilingual_fill import tag_spans, parse_tagged_to_segments  # (not required, just exposed)
from langdetect import detect
# ---- OpenAI translator wrapper ----
from openai import OpenAI

st.set_page_config(page_title="Bilingual Docx Maker (VI→EN)", page_icon="🌐", layout="centered")

st.title("🌐 Bilingual Document Maker (VI ⇄ EN)")
st.caption("Vietnamese first, then English. English styled as *italic* + Accent 1 (blue).")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Or set OPENAI_API_KEY env var.")
    add_labels = st.checkbox("Show [VI]/[EN] labels", value=True)
    table_mode = st.radio("Table mode", ["stack", "inline"], index=0)

    st.markdown("---")
    st.write("**Tips**")
    st.write("- Keep .docx (not .doc / PDF)")
    st.write("- Bold/Italic spans are preserved exactly")
    st.write("- Lists like '1.' or '-' are kept")

uploaded = st.file_uploader("Upload a .docx file", type=["docx"])

def make_openai_translator(key: str):
    client = OpenAI(api_key=key) if key else OpenAI()
    model = "gpt-4o-mini"

    def translate(text_with_tags: str, src_lang: str) -> str:
        target = "English" if src_lang == "vi" else "Vietnamese"
        sys = (
            "You are a professional translator. Translate the content while "
            "STRICTLY preserving these inline tags: <b>, </b>, <i>, </i>, <bi>, </bi>. "
            "Do not remove, add, or reorder tags; keep them around the same phrases they wrap. "
            "Preserve numbers, units, and proper nouns."
        )
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": sys},
                {"role": "user", "content": f"Translate to {target}:\n\n{text_with_tags}"}
            ]
        )
        return resp.output_text or ""
    return translate

if uploaded is not None:
    st.success("File received.")
    col1, col2 = st.columns(2)
    with col1:
        run_btn = st.button("Convert to Bilingual")
    with col2:
        sample_btn = st.button("Detect Language (quick check)")

    if sample_btn:
        data = uploaded.read()
        uploaded.seek(0)
        st.info(f"Detected language of first 1KB preview: **{detect(data[:1024].decode(errors='ignore'))}**")

    if run_btn:
        # Build translator
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            st.warning("Please enter your OpenAI API key (or set OPENAI_API_KEY env var).")
        else:
            translate_fn = make_openai_translator(key)
            # Write input to temp file
            with NamedTemporaryFile(delete=False, suffix=".docx") as in_tmp:
                in_tmp.write(uploaded.read())
                in_tmp.flush()
                in_path = in_tmp.name
            out_tmp = NamedTemporaryFile(delete=False, suffix=".docx")
            out_tmp.close()

            # Process
            with st.spinner("Translating and formatting..."):
                process_docx(
                    in_path=in_path,
                    out_path=out_tmp.name,
                    translate_fn=translate_fn,
                    add_labels=add_labels,
                    table_mode=table_mode
                )

            # Serve result
            with open(out_tmp.name, "rb") as f:
                st.download_button(
                    label="Download bilingual .docx",
                    data=f.read(),
                    file_name="output_bilingual.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            st.success("Done! Vietnamese first, English italic + Accent 1.")
