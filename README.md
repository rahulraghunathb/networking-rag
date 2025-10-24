# Networking RAG (OpenAI + Chroma)

Minimal Retrieval-Augmented Generation over local PDFs (network security slides) using:
- OpenAI embeddings: `text-embedding-3-large`
- Vector store: ChromaDB (persistent)
- LLM: `gpt-5-nano`

## Prerequisites
- Python 3.12+ (tested)
- An OpenAI API key

## Project Layout
- `ingest.py` — builds the Chroma index from PDFs under `data/docs/Network_Security_All_slides/`
- `query.py` — retrieves context and answers with GPT-5 Nano
- `chroma_store/` — Chroma persistence directory (auto-created)

## Setup (Windows PowerShell)
1) Create and activate a virtual environment
```
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

If you already have a virtual environment named `.rag`, activate it instead:
```
.\.rag\Scripts\activate
```

2) Install dependencies
```
pip install openai chromadb pypdf tiktoken python-dotenv
```

3) Configure environment
- Create a `.env` file in the project root with:
```
OPENAI_API_KEY=sk-...
```

## Ingestion
Build the vector index from PDFs.
- By default, PDFs are read from `data/docs/Network_Security_All_slides`.
- Embeddings: `text-embedding-3-large` (dimension 3072).

Optional: In `ingest.py`, set `reset_collection = True` for a clean rebuild. Then run:
```
python ingest.py
```
After a successful run, revert `reset_collection = False` to avoid wiping data on subsequent runs.

## Querying
Run the simple CLI, then type your question when prompted:
```
python query.py
```
The script:
- Embeds your query with `text-embedding-3-large`
- Retrieves top-k chunks from the Chroma collection `networking_slides`
- Calls `gpt-5-nano` to generate an answer, showing context blocks and sources (PDF name and page)

## Troubleshooting
- Missing collection error:
  - Error: `chromadb.errors.NotFoundError: Collection [networking_slides] does not exist`
  - Fix: Run ingestion first (`python ingest.py`). Ensure `chroma_store/` exists and is not empty.

- Embedding dimension mismatch:
  - Error like: `Collection expecting embedding dimension 3072, got 384`
  - Cause: Query used a different embedding than ingestion.
  - Fix: Ensure `query.py` uses OpenAI `text-embedding-3-large` (already configured as `DEFAULT_EMBED_MODEL`). Re-ingest if you previously indexed with a different model.

- PDF parsing warnings:
  - Some slides contain malformed objects; `ingest.py` uses `PdfReader(..., strict=False)` to suppress non-critical warnings.
  - If text can’t be selected in the PDF (handwritten or scanned), `pypdf.extract_text()` won’t read it. See OCR below.

## OCR for Handwritten/Scanned Slides (Optional)
`pypdf` reads only digital text. For image-only or handwritten content, add OCR before ingestion, e.g.:
- Tesseract (local, open-source)
- Azure Form Recognizer, Google Document AI, AWS Textract (managed)

You can preprocess PDFs to text with OCR and then adjust `ingest.py` to load from the OCR outputs (or integrate OCR in the ingestion loop).

## Configuration Defaults
- Collection name: `networking_slides`
- Chroma persist dir: `chroma_store`
- Chunking: approx 800 tokens with 200 stride (via `tiktoken`)
- Top-k: 4

## Notes
- Keep your API key secure; don’t commit `.env`.
- For large datasets, consider reranking, better chunking strategies, and evaluation tooling as next steps.
