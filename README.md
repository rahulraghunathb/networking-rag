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

- `api_main.py`: Main FastAPI server (entry point)
- `api_qa.py`: Q&A mode logic (question answering with context)
- `api_quiz.py`: Quiz mode logic (quiz generation and checking)
- `ingest_hf.py`: Build ChromaDB from PDFs
- `static/`: Frontend assets (CSS/JS)
- `templates/`: HTML UI

### Architecture

The application is now modularized:
- **api_main.py**: FastAPI app with endpoints
- **api_qa.py**: Handles Q&A mode with RAG
- **api_quiz.py**: Handles quiz generation with 20 hardcoded topics from the vector database

## Features

- **Q&A Mode**: Contextual answers with citations from the database
- **Enhanced Quiz Mode**: Interactive quizzes with multiple-choice, true/false, and open-ended questions
- **Question Generation**: Random or topic-specific questions pulled from the database with enhanced variety
- **Web Citations**: Additional references and citations from the internet for quiz feedback
- **Advanced Grading**: Confidence-based scoring with similarity analysis for open-ended questions
- **Modern UI**: Responsive design with animations and source tracking

## Usage

1. Add PDFs to `data/docs/pdfs`
2. Build index:

```bash
python ingest_hf.py
```

3. Start API server:

```bash
python api_main.py
```

4. Access UI: http://localhost:8000

### Q&A Mode

- Type questions in the input box
- Get answers with citations from the database
- Use example buttons for quick testing

### Enhanced Quiz Mode

1. Click "Quiz Mode" button in the header
2. Choose generation type:
   - **Random Questions**: Automatically selects from 20 hardcoded topics
   - **Topic-Specific Questions**: Select from dropdown of 20 networking topics
3. Select question type (Random, Multiple Choice Only, True/False Only, Open-Ended Only)
4. Set the number of questions (1-10)
5. Click "Generate Quiz" to get questions
6. Answer the questions and submit
7. View comprehensive feedback with:
   - Confidence-based grading (A-F scale)
   - Detailed explanations
   - Database citations from the knowledge base
   - Web citations from additional online sources
   - Similarity scoring for open-ended questions

**Hardcoded Topics** (20 total):
- Firewalls, DNS, TCP/IP Protocol, Network Security, Encryption/SSL/TLS
- VPN, DDoS Attacks, HTTP/HTTPS, Network Routing, OSI Model
- IP Addressing, Network Authentication, Wireless Security, IDS
- Load Balancing, Network Protocols, Packet Switching, Network Topology
- Cybersecurity Threats, Email Security

## Enhanced Quiz Examples

### Multiple Choice Questions
- **Beginner**: "What is the main purpose of TCP/IP in network communication?"
- **Intermediate**: "Which statement best describes how firewalls protect networks?"
- **Advanced**: "What are the implementation considerations for VPN security protocols?"

### True/False Questions
- **Beginner**: "True or False: HTTP is a secure protocol for data transmission."
- **Intermediate**: "True or False: DNS translates domain names to IP addresses."
- **Advanced**: "True or False: SSL certificates ensure end-to-end encryption."

### Open-Ended Questions
- **Beginner**: "What is a firewall and why is it important?"
- **Intermediate**: "Explain how encryption contributes to network security."
- **Advanced**: "Describe the technical mechanisms behind DDoS attack mitigation."

Questions are generated using enhanced algorithms with:
- Category-based question templates
- Context-aware distractors
- Difficulty-based complexity
- Comprehensive feedback with web citations

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
