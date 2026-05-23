"""
02_search.py — HyDE Search with Variant Testing
=================================================
Run this AFTER 01_ingest.py has completed.

What it does:
  1. Loads persisted Qdrant snapshot from DBFS (no re-embedding)
  2. Accepts a search term from the user
  3. Runs HyDE: Claude generates a hypothetical comment → embed → search
  4. Supports variant testing: try different HyDE prompts side-by-side
     without touching the index

Variant ideas you can test without re-ingesting:
  - Different HyDE prompt styles (clinical vs patient-voice vs neutral)
  - Different top_k values
  - Score thresholds
  - Month filters

Install in your Databricks cluster:
  %pip install anthropic sentence-transformers qdrant-client numpy pandas pyarrow

Set your API key before running:
  import os
  os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("your-scope", "anthropic-key")
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import anthropic
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Import shared config ──────────────────────────────────────────────
from config import (
    MONTH_COL, ID_COL, COMMENT_COL,
    EMBEDDINGS_PATH, METADATA_PATH, INGESTION_LOG_PATH,
    QDRANT_LOCAL_PATH, QDRANT_SNAPSHOT_PATH,
    COLLECTION_NAME, CLAUDE_MODEL, EMBED_MODEL_NAME,
    ANTHROPIC_API_KEY,
)


# ─────────────────────────────────────────────────────────────────────
# HYDE PROMPT VARIANTS
# Add / edit entries here to test different query expansion styles.
# Each variant is a prompt template with a {search_term} placeholder.
# ─────────────────────────────────────────────────────────────────────

HYDE_VARIANTS = {

    "patient_voice": """You are a healthcare survey analyst.
A user wants to find survey comments related to: "{search_term}"
Write ONE realistic 1-2 sentence patient comment expressing this concern,
as it would appear in an NPS survey. Use casual, conversational patient language.
Return ONLY the comment text, nothing else.""",

    "clinical_neutral": """You are a healthcare data analyst.
Generate a concise, neutral 1-2 sentence description of a patient experience
related to: "{search_term}"
Focus on the operational or clinical process involved.
Return ONLY the description, nothing else.""",

    "negative_framing": """You are a healthcare survey analyst.
A user is searching for complaints related to: "{search_term}"
Write ONE 1-2 sentence frustrated patient comment that would appear in a low NPS score survey.
Return ONLY the comment text, nothing else.""",

    "positive_framing": """You are a healthcare survey analyst.
A user is searching for positive feedback related to: "{search_term}"
Write ONE 1-2 sentence satisfied patient comment that would appear in a high NPS score survey.
Return ONLY the comment text, nothing else.""",

}

DEFAULT_VARIANT = "patient_voice"


# ─────────────────────────────────────────────────────────────────────
# SETUP: load clients and restore Qdrant from snapshot
# ─────────────────────────────────────────────────────────────────────

def load_ingestion_log() -> dict:
    if not Path(INGESTION_LOG_PATH).exists():
        raise FileNotFoundError(
            f"Ingestion log not found at {INGESTION_LOG_PATH}\n"
            "  Please run 01_ingest.py first."
        )
    with open(INGESTION_LOG_PATH) as f:
        return json.load(f)


def restore_qdrant() -> QdrantClient:
    """
    Restore Qdrant collection from DBFS snapshot into local temp path.
    This is fast — no re-embedding, just index reconstruction.
    """
    if not Path(QDRANT_SNAPSHOT_PATH).exists():
        raise FileNotFoundError(
            f"Qdrant snapshot not found at {QDRANT_SNAPSHOT_PATH}\n"
            "  Please run 01_ingest.py first."
        )

    restore_path = "/tmp/nps_qdrant_search"
    Path(restore_path).mkdir(parents=True, exist_ok=True)

    print(f"  Restoring Qdrant snapshot from DBFS ...")
    client = QdrantClient(path=restore_path)

    # If collection already exists from a previous run in this session, reuse it
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = client.get_collection(COLLECTION_NAME).points_count
        print(f"  ✓ Collection already loaded in session ({count} points)")
        return client

    # Otherwise restore from snapshot
    local_snap = f"/tmp/nps_restore_snap.snapshot"
    shutil.copy2(QDRANT_SNAPSHOT_PATH, local_snap)

    client.recover_snapshot(
        collection_name=COLLECTION_NAME,
        location=f"file://{local_snap}",
    )
    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"  ✓ Qdrant snapshot restored  ({count} points)")
    return client


# ─────────────────────────────────────────────────────────────────────
# HYDE — generate hypothetical comment for query
# ─────────────────────────────────────────────────────────────────────

def generate_hyde(
    claude: anthropic.Anthropic,
    search_term: str,
    variant: str = DEFAULT_VARIANT,
) -> str:
    """
    Generate a hypothetical patient comment for the search term
    using the specified prompt variant.
    """
    if variant not in HYDE_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(HYDE_VARIANTS)}")

    prompt = HYDE_VARIANTS[variant].format(search_term=search_term)

    resp = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ─────────────────────────────────────────────────────────────────────
# SEARCH — single variant run
# ─────────────────────────────────────────────────────────────────────

def search_comments(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    claude: anthropic.Anthropic,
    search_term: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,
    month_filter: Optional[str] = None,
    hyde_variant: str = DEFAULT_VARIANT,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run a single HyDE search and return results as a DataFrame.

    Parameters
    ----------
    search_term     : user's search phrase
    top_k           : number of results to return
    score_threshold : minimum cosine similarity (0–1); None = no threshold
    month_filter    : restrict to a specific month string (e.g. "2024-03")
    hyde_variant    : which HyDE prompt style to use (see HYDE_VARIANTS)
    verbose         : print HyDE comment to console
    """

    # Step 1: Generate hypothetical comment (HyDE)
    hypo_comment = generate_hyde(claude, search_term, variant=hyde_variant)
    if verbose:
        print(f"\n  HyDE [{hyde_variant}]:\n  → \"{hypo_comment}\"")

    # Step 2: Embed the hypothetical comment
    query_vec = embedder.encode(hypo_comment, normalize_embeddings=True).tolist()

    # Step 3: Optional filters
    search_filter = None
    if month_filter:
        search_filter = Filter(
            must=[FieldCondition(key="month", match=MatchValue(value=month_filter))]
        )

    # Step 4: Qdrant search
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        score_threshold=score_threshold,
        query_filter=search_filter,
        with_payload=True,
    )

    # Step 5: Build results DataFrame
    rows = []
    for rank, hit in enumerate(hits, start=1):
        p = hit.payload
        rows.append({
            "rank":        rank,
            "score":       round(hit.score, 4),
            "id":          p["original_id"],
            "month":       p["month"],
            "comment":     p["comment"],
            "d2q_tags":    " | ".join(p.get("doc2query", [])),
            "hyde_variant": hyde_variant,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# VARIANT COMPARISON — run multiple HyDE styles side by side
# ─────────────────────────────────────────────────────────────────────

def compare_variants(
    qdrant: QdrantClient,
    embedder: SentenceTransformer,
    claude: anthropic.Anthropic,
    search_term: str,
    variants: Optional[list[str]] = None,
    top_k: int = 5,
    month_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run multiple HyDE prompt variants for the same search term
    and return a single DataFrame for side-by-side comparison.
    """
    variants = variants or list(HYDE_VARIANTS.keys())
    all_results = []

    for variant in variants:
        print(f"\n  ── Variant: {variant} ──")
        df = search_comments(
            qdrant, embedder, claude,
            search_term=search_term,
            top_k=top_k,
            month_filter=month_filter,
            hyde_variant=variant,
            verbose=True,
        )
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def display_results(df: pd.DataFrame, search_term: str, group_by_variant: bool = False):
    """Pretty-print results. If group_by_variant, print each variant separately."""
    if df.empty:
        print("  No results found.")
        return

    def _print_block(block: pd.DataFrame, label: str):
        print(f"\n  {'─'*60}")
        print(f"  {label}  |  {len(block)} result(s)")
        print(f"  {'─'*60}")
        for _, row in block.iterrows():
            print(f"  #{int(row['rank'])}  score={row['score']}  "
                  f"id={row['id']}  month={row['month']}")
            print(f"      comment : {row['comment']}")
            print(f"      d2q tags: {row['d2q_tags']}")

    if group_by_variant and "hyde_variant" in df.columns:
        for variant, group in df.groupby("hyde_variant"):
            _print_block(group, f"Variant: {variant}")
    else:
        _print_block(df, f'Results for: "{search_term}"')

    print()


# ─────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Startup ───────────────────────────────────────────────────────
    print("=" * 65)
    print("  NPS Comment Search  |  HyDE + Doc2Query")
    print("=" * 65)

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set.\n"
            "  os.environ['ANTHROPIC_API_KEY'] = dbutils.secrets.get('scope', 'key')"
        )

    log = load_ingestion_log()
    print(f"\n  Index info:")
    print(f"    Ingested at  : {log['timestamp']}")
    print(f"    Comments     : {log['row_count']}")
    print(f"    Embed model  : {log['embed_model']}")
    print(f"    Claude model : {log['claude_model']}")
    print(f"    Vector dim   : {log['vector_dim']}")

    qdrant_client = restore_qdrant()
    embedder      = SentenceTransformer(EMBED_MODEL_NAME)
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    print(f"\n  Available HyDE variants: {list(HYDE_VARIANTS.keys())}")
    print("  Enter 'compare' to run all variants side by side.")
    print("  Press Enter with no search term to exit.\n")

    # ── Interactive search loop ───────────────────────────────────────
    while True:
        print("─" * 65)
        search_term = input("  Search term          : ").strip()
        if not search_term:
            print("  Exiting. Goodbye!")
            break

        mode = input(
            "  Mode — single variant or compare all? [single/compare] (default: single): "
        ).strip().lower() or "single"

        month_input = input(f"  Filter by month (e.g. 2024-03, blank = all)  : ").strip()
        month_filter = month_input or None

        top_k_input = input("  Number of results per variant (default 5)    : ").strip()
        top_k = int(top_k_input) if top_k_input.isdigit() else 5

        threshold_input = input("  Min score threshold (e.g. 0.4, blank = none) : ").strip()
        score_threshold = float(threshold_input) if threshold_input else None

        print()

        if mode == "compare":
            results = compare_variants(
                qdrant_client, embedder, claude_client,
                search_term=search_term,
                top_k=top_k,
                month_filter=month_filter,
            )
            display_results(results, search_term, group_by_variant=True)

        else:
            variant_input = input(
                f"  HyDE variant {list(HYDE_VARIANTS.keys())} (default: {DEFAULT_VARIANT}): "
            ).strip() or DEFAULT_VARIANT

            results = search_comments(
                qdrant_client, embedder, claude_client,
                search_term=search_term,
                top_k=top_k,
                score_threshold=score_threshold,
                month_filter=month_filter,
                hyde_variant=variant_input,
            )
            display_results(results, search_term)

        # ── Optional: surface as Spark DataFrame in Databricks ────────
        # spark_df = spark.createDataFrame(results)
        # display(spark_df)
