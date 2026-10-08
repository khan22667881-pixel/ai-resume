# ai-resume
# 📄 ATS Resume Checker

Upload a resume (PDF, DOCX or TXT), optionally paste a job description, and get an
**ATS score out of 100**, a score breakdown, missing keywords, prioritised
improvements and example bullet rewrites. Built with **Streamlit** and **Google Gemini Flash**.

## Features
- PDF / DOCX / TXT resume upload
- Optional job description for targeted keyword matching
- Score breakdown: keywords, content & impact, structure, formatting, readability
- Rule-based checks (email, phone, LinkedIn, sections, quantified achievements)
- Prioritised improvements and before/after bullet rewrites
- Download the report as JSON

## How the score works
Gemini rates five categories from 0 to 100. The overall score is then calculated in
code with fixed weights (keywords 30%, content 30%, structure 15%, formatting 15%,
readability 10%) so results are consistent. It is an **estimate**, not the output of
a real employer's ATS.

## Run locally
```bash
git clone https://github.com/<your-username>/ats-resume-checker.git
cd ats-resume-checker
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then either paste it in
the app sidebar, or set it once:

```bash
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

Or create `.streamlit/secrets.toml` (never commit this file):
```toml
GEMINI_API_KEY = "your-key"
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (public, or private with access granted).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
5. Click **Deploy**.

## Configuration
| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets / env var / sidebar | none |
| Model name | Sidebar | `gemini-2.5-flash` |

## Privacy
Resume text is sent to the Gemini API for analysis and is not stored by this app.

## Project structure
```
app.py             # Streamlit app + analysis logic
requirements.txt   # Python dependencies
README.md
```

## Limitations
- Scanned/image-only PDFs have no extractable text and can't be analysed.
- AI scoring is an approximation; use the suggestions as guidance.
