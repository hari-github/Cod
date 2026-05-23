"""
config.py — Shared configuration for Doc2Query + HyDE Search Pipeline
======================================================================
Edit this file once. Both 01_ingest.py and 02_search.py import from here.
"""

import os

# ─────────────────────────────────────────────────────────────────────
# 1. COLUMN NAMES
#    Set these to match your actual DataFrame column names exactly.
# ─────────────────────────────────────────────────────────────────────

MONTH_COL   = "Month"      # e.g. "survey_month", "period", "date"
ID_COL      = "ID"         # e.g. "response_id", "record_id", "survey_id"
COMMENT_COL = "Comment"    # e.g. "verbatim", "nps_comment", "feedback_text"


# ─────────────────────────────────────────────────────────────────────
# 2. STORAGE PATHS (DBFS)
#    All persisted artifacts land here. Change the root to your mount.
# ─────────────────────────────────────────────────────────────────────

STORE_ROOT          = "/dbfs/tmp/nps_search"          # ← change to your DBFS path

EMBEDDINGS_PATH     = f"{STORE_ROOT}/ingestion/embeddings.npy"
METADATA_PATH       = f"{STORE_ROOT}/ingestion/metadata.parquet"
INGESTION_LOG_PATH  = f"{STORE_ROOT}/ingestion/ingestion_log.json"
QDRANT_SNAPSHOT_DIR = f"{STORE_ROOT}/qdrant"
QDRANT_SNAPSHOT_PATH= f"{QDRANT_SNAPSHOT_DIR}/nps_comments.snapshot"

# Local (non-DBFS) Qdrant storage path — Qdrant needs a real local path, not /dbfs/
# Databricks driver node local disk; survives the session but not cluster restart
QDRANT_LOCAL_PATH   = "/tmp/nps_qdrant"


# ─────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────

CLAUDE_MODEL        = "claude-sonnet-4-20250514"
EMBED_MODEL_NAME    = "BAAI/bge-small-en-v1.5"   # 384-dim, fast, good on short text
N_DOC2QUERY         = 5                           # queries generated per comment
BATCH_DELAY_SECS    = 0.3                         # delay between Claude API calls


# ─────────────────────────────────────────────────────────────────────
# 4. QDRANT
# ─────────────────────────────────────────────────────────────────────

COLLECTION_NAME     = "nps_comments"


# ─────────────────────────────────────────────────────────────────────
# 5. API KEY
# ─────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


# ─────────────────────────────────────────────────────────────────────
# 6. SAMPLE DATA  (used when no real data is provided)
# ─────────────────────────────────────────────────────────────────────

SAMPLE_DATA = {
    MONTH_COL: [
        "2024-01", "2024-01", "2024-02", "2024-02", "2024-03",
        "2024-03", "2024-03", "2024-04", "2024-04", "2024-05",
        "2024-05", "2024-06", "2024-06", "2024-06", "2024-07",
    ],
    ID_COL: [f"C{str(i).zfill(4)}" for i in range(1, 16)],
    COMMENT_COL: [
        "Getting approvals is a nightmare, took three weeks.",
        "The app keeps crashing when I try to view my claims.",
        "My doctor said the referral was denied without explanation.",
        "Billing statement was confusing and had wrong amounts.",
        "Had to call five times before someone resolved my issue.",
        "Prescription coverage was unclear on the website.",
        "Prior auth process is way too slow and frustrating.",
        "Easy to schedule an appointment online, very smooth.",
        "Claim reimbursement took two months, completely unacceptable.",
        "Customer service was helpful and resolved it quickly.",
        "Finding an in-network specialist was really difficult.",
        "The member portal is hard to navigate and outdated.",
        "I was denied coverage for a procedure my doctor recommended.",
        "Copay amounts keep changing without any notification.",
        "Overall satisfied with the plan, renewal was straightforward.",
    ],
}
