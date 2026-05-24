# ScholarSync MCP Academic Server

[![Tests](https://github.com/Shaurya000-png/ScholarSync-MCP-Academic-Server/actions/workflows/test.yml/badge.svg)](https://github.com/Shaurya000-png/ScholarSync-MCP-Academic-Server/actions/workflows/test.yml)

A powerful Python Model Context Protocol (MCP) server that acts as an AI-powered academic assistant. ScholarSync connects Claude Desktop (or any MCP client) to a student's academic data stored in MongoDB. It features a complete semantic RAG pipeline for querying notes, an advanced PDF processing engine, timetable tracking, and an intelligent task/assignment manager.

## Features

- **Semantic RAG Pipeline:** Ingest your notes and PDFs into semantic chunks. Supports Hybrid Search (BM25 + Vector) using Reciprocal Rank Fusion (RRF).
- **Advanced PDF Extraction:** Extracts text and tables, with an OCR fallback (using pytesseract) for scanned documents, and auto-detects subjects using heuristic rules.
- **Academic Management:** Full CRUD capabilities for assignments, tasks, and scheduling directly from Claude.
- **Multilingual Support:** Ask questions in any language (e.g., Hindi, Marathi) and it will be translated, searched over the English knowledge base, and returned.
- **Proactive Insights:** Claude can automatically warn you about urgent deadlines and identify knowledge gaps in your study materials.
- **Docker Ready:** Easily spin up the entire application stack including MongoDB with Docker Compose.

## Tech Stack

- **Python 3.11+**
- **Anthropic MCP SDK** (`mcp` / `FastMCP`)
- **MongoDB** (Atlas or local via PyMongo)
- **sentence-transformers** for local embeddings
- **PyMuPDF / PyPDF2 / pytesseract** for PDF and Table extraction
- **rank_bm25 & googletrans** for Hybrid & Multilingual search
- **Pydantic v2** for validation
- **pytest** for comprehensive testing

## Installation

### Option 1: Docker Compose (Recommended)

1. Clone the repository.
2. Spin up the application and MongoDB using Docker Compose:
   ```bash
   docker-compose up -d --build
   ```

### Option 2: Local Setup

1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the `.env.example` file to `.env` and fill in your variables. By default, it expects a local MongoDB instance.

## Claude Desktop Configuration

Add the server to your Claude Desktop MCP configuration file (typically `claude_desktop_config.json`). Make sure to adjust the path to wherever you cloned this repository.

```json
{
  "mcpServers": {
    "scholarsync-mcp-academic-server": {
      "command": "python",
      "args": [
        "/absolute/path/to/ScholarSync-MCP-Academic-Server/server.py"
      ],
      "env": {
        "MONGODB_URI": "mongodb://localhost:27017",
        "MONGODB_DATABASE": "academic_assistant"
      }
    }
  }
}
```

## Available MCP Tools

ScholarSync exposes **20** powerful tools for Claude to utilize:

**Retrieval & Search**
- `search_notes` - Keyword search across MongoDB and local notes.
- `retrieve_academic_context` - Semantic context retrieval for question answering.
- `semantic_search_academic_knowledge` - Run semantic/hybrid searches.
- `translate_and_search` - Multilingual querying capability.

**Ingestion & PDFs**
- `ingest_academic_knowledge_base` - Process and embed notes and PDFs.
- `summarize_pdf` - Extract, read, and summarize local PDF files.

**Academic Tracking**
- `search_assignments`, `add_assignment`, `update_assignment_status`
- `add_task`, `view_tasks`, `mark_task_complete`
- `get_schedule`, `get_subject_info`

**Advanced Planning & Generation**
- `generate_exam_questions` - Automatically generate practice questions (MCQs, T/F) based on notes.
- `create_study_plan` - Generates a study schedule using upcoming deadlines and timetable data.
- `find_knowledge_gaps` - Highlights subjects with assignments but missing study notes.
- `get_urgent_items` - Retrieves items due within the next N hours.

**System Utilities**
- `query_student_database` - Safe, direct exact-match querying.
- `health_check` - Ensures connectivity to MongoDB.

## Testing

The project has robust test coverage (41+ tests).
Run the test suite with:

```bash
python -m pytest -v
```

## License

This project is open-source and available under the MIT License.
