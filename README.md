# NovaPay HR Policy & Benefits Assistant

A robust, enterprise-grade Retrieval-Augmented Generation (RAG) assistant built to answer employee questions strictly and accurately from the official 50-page **NovaPay Global Human Resources, Compensation & Benefits Manual**.

---

## 🏛️ Architecture Overview

The assistant follows an end-to-end grounded RAG pipeline:

```
[ novapay_global_hr_manual_50p.pdf (50 Pages) ]
                    │
                    ▼
          [ PyPDF Extraction ]
      (Preserves page 1-50 metadata,
       filters empty/corrupt pages)
                    │
                    ▼
     [ Recursive Character Chunking ]
   (Chunk Size: 700, Overlap: 120,
    Tags explicit SECTION & policy lines)
                    │
                    ▼
         [ Gemini / Offline Embeddings ]
    (Google text-embedding-004 or
     deterministic normalized embeddings)
                    │
                    ▼
          [ FAISS Vector Store ]
                    │
           ┌────────┴────────┐
           │ User Query      │
           ▼                 ▼
   [ Vector Similarity Search & Re-ranking ]
   (Retrieves top-k chunks, boosts substantive
    clauses over repetitive boilerplate)
                    │
                    ▼
      [ Grounded Gemini LLM Synthesis ]
      (Strict adherence to context, source
       citations [Page X], strict fallback:
       "This information was not found in the provided HR manual")
                    │
                    ▼
       [ Streamlit Interactive Chat UI ]
```

---

## 🚀 Key Features

1. **Strict Context Grounding**:
   - The LLM will **never invent missing policy information**.
   - Answers exclusively from retrieved text and provides exact page citations (e.g. `[Page 11]`, `[Page 39]`).
   - If a topic is absent or only mentioned as a generic module header, the system reliably outputs:
     `"This information was not found in the provided HR manual."`

2. **Metadata Integrity**:
   - Every single text chunk retains its exact 1-indexed document page number, source document name, and content classifications.

3. **Intelligent Boilerplate Handling**:
   - The NovaPay HR manual contains repeated procedural text across various modules alongside critical explicit sections (such as Band-3 ESOP vesting on Page 11, international travel lodging caps on Page 27, and paternity leave on Page 39).
   - The retriever uses content-aware scoring to prioritize explicit policy parameters over repetitive administrative boilerplate.

4. **Dual Embedding & Inference Mode**:
   - **Gemini API Mode**: When `GEMINI_API_KEY` is provided, the system leverages Google's `models/text-embedding-004` and `gemini-1.5-flash` for high-precision semantic retrieval and synthesis.
   - **Offline Deterministic Mode**: Zero external dependencies required for testing or air-gapped environments. FAISS indexing and policy extraction work offline.

5. **Modern Streamlit Interface**:
   - NovaPay enterprise branding with dark/light mode compatibility.
   - Dynamic PDF upload & auto-indexing.
   - Interactive chat history with collapsible source citation accordions.
   - Pre-configured quick-query buttons for testing.
   - Masked API key configuration with status indicators.

---

## 📂 Project Structure

```
genaiexam/
├── app.py                     # Streamlit frontend & interactive UI
├── rag_pipeline.py            # Complete RAG pipeline (extract, chunk, embed, FAISS, retrieve, generate)
├── requirements.txt           # Verified dependencies
├── .env.example               # Environment template
├── .gitignore                 # Standard Python/Streamlit/Venv gitignore
├── README.md                  # System documentation
├── data/
│   └── novapay_global_hr_manual_50p.pdf  # 50-page official HR manual
├── tests/
│   └── test_rag.py            # Comprehensive automated test suite (Tests A - I)
└── vectorstore/               # Persisted FAISS index files
```

---

## 🛠️ Installation & Setup

### 1. Clone & Activate Virtual Environment
```bash
git clone https://github.com/Babul422/genaiexam.git
cd genaiexam

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and configure your Gemini API Key:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, enter your API key directly in the Streamlit sidebar during runtime).*

---

## 🧪 Running Automated Tests

Run the automated test suite covering all 9 assessment criteria (Imports, PDF extraction, Chunking, FAISS index, Retrieval, Citations, Fallback generation, Error handling):

```bash
pytest tests/test_rag.py -v
```

---

## 🖥️ Running the Streamlit Application

Start the web application locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
