# AcademicAdvisor

A natural-language assistant for exploring Carleton's graduate course catalog
and academic policies. Built to demonstrate agentic AI patterns over mixed
structured (PostgreSQL) and unstructured (RAG) data sources.

**Status:** in development. This README will grow as the project does.

## What it does (planned)

- **Text-to-SQL:** Ask "Which winter-term COMP courses have no prerequisites?"
  and the agent generates and runs PostgreSQL against the course catalog.
- **RAG over policies:** Ask "How many credits do I need for the MEng?" and the
  agent retrieves the relevant section of the graduate calendar before answering.
- **Routing:** A LangGraph agent decides which tool to use based on the question.
- **Guardrails:** Generated SQL is validated against an allow-list before execution;
  low-confidence retrievals are flagged rather than fabricated.

## Architecture (Day 1)

- PostgreSQL 16 with `pgvector` for both relational and vector data
- Single-file schema in `src/db/schema.sql`
- 30 real Carleton graduate courses seeded from `data/seed/courses.csv`

## Quick start

```bash
docker compose up -d
cp .env.example .env
python -m src.db.seed
python -m src.db.check
```

## Roadmap

- [x] Day 1: PostgreSQL + pgvector + seeded courses
- [ ] Day 2: Document ingestion (graduate calendar excerpts) + embeddings
- [ ] Day 3: Retrieval pipeline with confidence scoring
- [ ] Day 4: Text-to-SQL tool + SQL allow-list guardrail
- [ ] Day 5: LangGraph agent routing between SQL and RAG tools
- [ ] Day 6: FastAPI wrapper + Docker compose for the full stack
- [ ] Day 7: Evaluation harness with 20 test queries

## Design decisions

- **pgvector over FAISS/Chroma:** single-database deployment, transactional
  consistency between structured course data and vectors, native HNSW indexing.
- **Generated `has_prereqs` column:** removes a class of LLM hallucination by
  having the database compute derived booleans at write time.
- **`HNSW` index:** no training step required, good recall on small-to-medium
  corpora.
- **`query_log` table:** foundation for the evaluation framework — every agent
  run is replayable post-hoc without re-running the LLM.

## Tech stack

Python 3.11 · PostgreSQL 16 + pgvector · LangChain · LangGraph · Anthropic Claude · sentence-transformers · FastAPI · Docker
