-- AcademicAdvisor schema
-- Designed for plain-language queries over Carleton graduate course data

CREATE EXTENSION IF NOT EXISTS vector;

-- Courses: structured data the agent will Text-to-SQL over
CREATE TABLE IF NOT EXISTS courses (
    id              SERIAL PRIMARY KEY,
    course_code     TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    department      TEXT NOT NULL,
    level           INT NOT NULL CHECK (level BETWEEN 5000 AND 6999),
    credits         NUMERIC(2, 1) NOT NULL CHECK (credits >= 0),
    description     TEXT,
    prerequisites   TEXT,
    has_prereqs     BOOLEAN GENERATED ALWAYS AS (
        prerequisites IS NOT NULL AND prerequisites != ''
    ) STORED,
    terms_offered   TEXT[] NOT NULL,
    instructor      TEXT,
    is_thesis       BOOLEAN DEFAULT FALSE,
    is_project      BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_courses_department ON courses(department);
CREATE INDEX IF NOT EXISTS idx_courses_level ON courses(level);
CREATE INDEX IF NOT EXISTS idx_courses_terms ON courses USING GIN(terms_offered);

-- RAG corpus: chunks of graduate calendar, MEng handbook, academic regulations
CREATE TABLE IF NOT EXISTS doc_chunks (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(384),
    metadata    JSONB DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding ON doc_chunks
    USING hnsw (embedding vector_cosine_ops);

-- Query log: enables evaluation and observability
CREATE TABLE IF NOT EXISTS query_log (
    id              SERIAL PRIMARY KEY,
    user_query      TEXT NOT NULL,
    tool_used       TEXT,
    generated_sql   TEXT,
    confidence      REAL,
    success         BOOLEAN,
    error_message   TEXT,
    latency_ms      INT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
