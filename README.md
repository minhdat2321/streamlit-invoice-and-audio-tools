# AI Tools: Transcribe & Invoice Extract (Streamlit)

A Streamlit app exposing two practical tools:

- **Transcribe audio** to TXT/SRT/VTT
- **Extract invoice images** into a structured table (CSV/XLSX)

## Demo Screenshots
*(Add your own after running locally)*

## Quickstart

### 1) Clone & install
```bash
git clone https://github.com/<your-username>/streamlit-invoice-and-audio-tools.git
cd streamlit-invoice-and-audio-tools
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Add your OpenAI API key
- **Local env**: create a `.env` or `.streamlit/secrets.toml` (preferred for Streamlit). Example:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

### 3) Run
```bash
streamlit run streamlit_app.py
```

Open the URL shown by the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. On https://share.streamlit.io/ → New app → point to `streamlit_app.py`.
3. In the app’s **Secrets** panel, paste your `OPENAI_API_KEY` as TOML.

## Security notes
- **Never commit API keys.** Use Streamlit secrets or environment variables.
- The app performs client‑side uploads to the Streamlit server where the OpenAI calls are made server‑side.