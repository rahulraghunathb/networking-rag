# Networking RAG System

Professional Q&A system for networking topics using:

- **Retrieval**: HuggingFace SentenceTransformers (msmarco-distilbert-base-v4)
- **Vector Store**: ChromaDB
- **Generation**: OpenAI GPT-5 Nano
- **API**: FastAPI with structured output
- **UI**: Modern chat interface

## Quick Start

1. Activate virtual environment:

```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install requirements:

```bash
pip install fastapi uvicorn chromadb sentence-transformers openai python-dotenv pypdf
```

3. Set OpenAI key in `.env`:

```
OPENAI_API_KEY=your_key
```

4. Build the knowledge base (run once or when adding new PDFs):

```bash
python ingest_hf.py
```

5. Run API (includes web UI):

```bash
python api_hf.py
```

6. Open UI: http://localhost:8000

## Key Files to Run

- **ingest_hf.py**: Run to build/update the ChromaDB vector store from PDFs
- **api_hf.py**: Run to start the FastAPI server and serve the web UI

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- ChromaDB: Vector database
- SentenceTransformers: HuggingFace embeddings
- OpenAI: GPT-5 Nano for summarization
- Python-dotenv: Environment variables
- PyPDF: PDF processing

## Key Files

- `api_hf.py`: Main API server
- `ingest_hf.py`: Build ChromaDB from PDFs
- `static/`: Frontend assets (CSS/JS)
- `templates/`: HTML UI

## Usage

1. Add PDFs to `data/docs/pdfs`
2. Build index:

```bash
python ingest_hf.py
```

3. Start API and chat at http://localhost:8000

## System Architecture

```mermaid
graph TD
    A[User Query] --> B[API (FastAPI)]
    B --> C[HF Embeddings]
    C --> D[ChromaDB Retrieval]
    D --> E[Context Extraction]
    E --> F[OpenAI GPT-5 Nano]
    F --> G[Generate Answer]
    G --> H[Return Response with Citations]
    H --> I[Web UI Display]
```

### Flow Description

1. **User Query**: Input via web UI or API
2. **HF Embeddings**: Convert query to vector using SentenceTransformers
3. **ChromaDB Retrieval**: Find top-k similar chunks
4. **Context Extraction**: Gather relevant text and metadata
5. **OpenAI GPT-5 Nano**: Summarize context into coherent answer
6. **Response**: JSON with answer, citations, and sources
7. **UI Display**: Render in modern chat interface

## How We Built It (Step-by-Step)

1. **Setup Project Structure**

   - Created directories: `data/docs/pdfs`, `static/`, `templates/`
   - Set up virtual environment and installed dependencies

2. **Data Ingestion (ingest_hf.py)**

   - Used PyPDF2 to read PDFs from `data/docs/pdfs`
   - Chunked text into 800-token segments with 200-token stride
   - Generated embeddings with HuggingFace SentenceTransformers
   - Stored in ChromaDB collection `networking_context`

3. **API Development (api_hf.py)**
   - Built FastAPI server with endpoints: `/`, `/health`, `/ask`
   - Integrated retrieval: Embed query → Search ChromaDB → Get contexts
   - Added OpenAI summarization for answers
   - Included CORS for cross-origin requests
