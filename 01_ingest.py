"""
01_ingest.py — Doc2Query Ingestion Pipeline
============================================
Run this ONCE (or when source data changes).

What it does:
  1. Loads your NPS comment DataFrame
  2. Calls Claude to generate Doc2Query expansions for each comment
  3. Embeds the expanded text using a sentence transformer
  4. Persists three artifacts to DBFS:
       embeddings.npy        ← raw numpy vectors  (skip re-embedding on reload)
       metadata.parquet      ← id, month, comment, doc2query tags
       ingestion_log.json    ← run metadata (timestamp, model, row count)
  5. Builds a Qdrant collection from persisted artifacts
  6. Saves a Qdrant snapshot to DBFS for fast reload in 02_search.py

Install in your Databricks cluster:
  %pip install anthropic sentence-transformers qdrant-client numpy pandas pyarrow

Set your API key before running:
  import os
  os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("your-scope", "anthropic-key")
"""

import os
import json
import time
import shutil
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import anthropic
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Import shared config ──────────────────────────────────────────────
from config import (
    MONTH_COL, ID_COL, COMMENT_COL,
    EMBEDDINGS_PATH, METADATA_PATH, INGESTION_LOG_PATH,
    QDRANT_LOCAL_PATH, QDRANT_SNAPSHOT_DIR, QDRANT_SNAPSHOT_PATH,
    STORE_ROOT, COLLECTION_NAME,
    CLAUDE_MODEL, EMBED_MODEL_NAME,
    N_DOC2QUERY, BATCH_DELAY_SECS,
    ANTHROPIC_API_KEY, SAMPLE_DATA,
)

# ─────────────────────────────────────────────────────────────────────
# HELPERS: directory setup
# ─────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for path in [
        f"{STORE_ROOT}/ingestion",
        QDRANT_SNAPSHOT_DIR,
        QDRANT_LOCAL_PATH,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# DOC2QUERY — LLM expansion
# ─────────────────────────────────────────────────────────────────────

def generate_doc2query(client: anthropic.Anthropic, comment: str) -> list[str]:
    """
    Ask Claude to generate N short search phrases a user might type
    to find this comment. Returns list of phrase strings.
    """
    prompt = f"""You are a healthcare survey analyst.
A patient wrote this NPS comment: "{comment}"

Generate exactly {N_DOC2QUERY} short search phrases (2-5 words each) that someone
might type to find this comment. Focus on the core healthcare issue expressed.

Return ONLY a JSON array of strings, no explanation. Example:
["prior auth delay", "approval wait time", "treatment denied"]"""

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    try:
        phrases = json.loads(raw)
        return [p.strip() for p in phrases if isinstance(p, str)]
    except json.JSONDecodeError:
        return []   # original comment still gets indexed


def build_expanded_text(comment: str, queries: list[str]) -> str:
    """Join original comment + generated queries into one embeddable string."""
    return " | ".join([comment] + queries)


# ─────────────────────────────────────────────────────────────────────
# INGESTION — main pipeline
# ─────────────────────────────────────────────────────────────────────

def run_ingestion(df: pd.DataFrame):
    """
    Full ingestion pipeline:
      Doc2Query → embed → persist numpy + parquet → build Qdrant → snapshot
    """
    ensure_dirs()

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "  os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get('scope', 'key')"
        )

    claude  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    vector_dim = embedder.get_sentence_embedding_dimension()

    total = len(df)
    print("=" * 65)
    print(f"  INGESTION  |  {total} comments  |  model: {EMBED_MODEL_NAME}")
    print("=" * 65)

    records      = []   # metadata rows
    vectors_list = []   # numpy vectors

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        comment   = str(row[COMMENT_COL])
        record_id = str(row[ID_COL])
        month     = str(row[MONTH_COL])

        print(f"\n  [{idx}/{total}]  {record_id}  |  {comment[:65]}...")

        # ── Step 1: Doc2Query via Claude ──────────────────────────────
        d2q_phrases = generate_doc2query(claude, comment)
        print(f"  Doc2Query → {d2q_phrases}")

        # ── Step 2: Build expanded text & embed ───────────────────────
        expanded = build_expanded_text(comment, d2q_phrases)
        vector   = embedder.encode(expanded, normalize_embeddings=True)

        vectors_list.append(vector)
        records.append({
            "original_id":    record_id,
            "month":          month,
            "comment":        comment,
            "doc2query":      json.dumps(d2q_phrases),   # store as JSON string in parquet
            "expanded_text":  expanded,
            "qdrant_id":      abs(hash(record_id)) % (2 ** 31),
        })

        time.sleep(BATCH_DELAY_SECS)

    # ── Step 3: Persist embeddings (numpy) ───────────────────────────
    vectors_np = np.array(vectors_list, dtype=np.float32)
    np.save(EMBEDDINGS_PATH, vectors_np)
    print(f"\n  ✓ Saved embeddings  → {EMBEDDINGS_PATH}  shape={vectors_np.shape}")

    # ── Step 4: Persist metadata (parquet) ───────────────────────────
    meta_df = pd.DataFrame(records)
    meta_df.to_parquet(METADATA_PATH, index=False)
    print(f"  ✓ Saved metadata    → {METADATA_PATH}  rows={len(meta_df)}")

    # ── Step 5: Write ingestion log ───────────────────────────────────
    log = {
        "timestamp":     datetime.datetime.utcnow().isoformat() + "Z",
        "claude_model":  CLAUDE_MODEL,
        "embed_model":   EMBED_MODEL_NAME,
        "n_doc2query":   N_DOC2QUERY,
        "row_count":     total,
        "vector_dim":    int(vector_dim),
        "columns":       {
            "month":   MONTH_COL,
            "id":      ID_COL,
            "comment": COMMENT_COL,
        },
    }
    with open(INGESTION_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓ Saved ingestion log → {INGESTION_LOG_PATH}")

    # ── Step 6: Build Qdrant collection ──────────────────────────────
    print(f"\n  Building Qdrant collection at {QDRANT_LOCAL_PATH} ...")
    qdrant = QdrantClient(path=QDRANT_LOCAL_PATH)

    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )

    points = [
        PointStruct(
            id=rec["qdrant_id"],
            vector=vectors_np[i].tolist(),
            payload={
                "original_id":   rec["original_id"],
                "month":         rec["month"],
                "comment":       rec["comment"],
                "doc2query":     json.loads(rec["doc2query"]),
                "expanded_text": rec["expanded_text"],
            },
        )
        for i, rec in enumerate(records)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  ✓ Qdrant collection built  ({len(points)} points)")

    # ── Step 7: Save Qdrant snapshot to DBFS ─────────────────────────
    # Qdrant snapshots land in <qdrant_local_path>/collection/<name>/snapshots/
    snapshot_info = qdrant.create_snapshot(collection_name=COLLECTION_NAME)
    local_snapshot = (
        Path(QDRANT_LOCAL_PATH)
        / "collection"
        / COLLECTION_NAME
        / "snapshots"
        / snapshot_info.name
    )
    shutil.copy2(str(local_snapshot), QDRANT_SNAPSHOT_PATH)
    print(f"  ✓ Qdrant snapshot saved → {QDRANT_SNAPSHOT_PATH}")

    print("\n" + "=" * 65)
    print("  INGESTION COMPLETE")
    print(f"  Artifacts saved under: {STORE_ROOT}")
    print("  Run 02_search.py to start searching.")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load your data ─────────────────────────────────────────────────
    # Option A: Use sample data (default for testing)
    df = pd.DataFrame(SAMPLE_DATA)

    # Option B: Load from Spark table
    # df = spark.table("hive_metastore.default.nps_comments").toPandas()

    # Option C: Load from DBFS CSV
    # df = pd.read_csv("/dbfs/mnt/your-mount/nps_comments.csv")

    # Validate columns exist
    for col_label, col_name in [("Month", MONTH_COL), ("ID", ID_COL), ("Comment", COMMENT_COL)]:
        if col_name not in df.columns:
            raise ValueError(
                f"Column '{col_name}' not found in DataFrame.\n"
                f"  Available columns: {list(df.columns)}\n"
                f"  Update {col_label}_COL in config.py"
            )

    print(f"\n  Loaded DataFrame: {len(df)} rows")
    print(f"  Columns → Month: '{MONTH_COL}'  ID: '{ID_COL}'  Comment: '{COMMENT_COL}'\n")

    run_ingestion(df)
