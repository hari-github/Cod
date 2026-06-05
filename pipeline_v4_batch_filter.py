# =============================================================================
# pipeline_v1_batch_filter.py
#
# ARCHITECTURE: Batch LLM Tag Extraction + Dual-Stream Tag Search
#
# Ingest phase:
#   - Claude extracts tags only (no intents) in batches of 5
#   - Tags are expanded — no abbreviations, full clinical terms
#   - Each tag is embedded individually (exploded index pattern)
#     so search is always short phrase vs short phrase
#   - Full-text index built on tags for exact token matching
#
# Query phase — two separate ranked streams:
#   Stream A | Context search
#             Query phrase expanded to synonyms via Claude,
#             each synonym embedded and searched independently,
#             results merged by doc_id keeping best tag score
#
#   Stream B | Exact tag search
#             Full-text token match on stored tag strings,
#             word-level tokenisation, case-normalised
#
#   Overlap between streams flagged explicitly in output.
#
# Auth:
#   Uses OpenAI-compatible Databricks client for both LLM and embeddings.
#   DATABRICKS_BASE_URL, LLM_MODEL, EMBED_MODEL collected at startup.
# =============================================================================

# %pip install openai "qdrant-client>=1.9.0" pydantic
# dbutils.library.restartPython()


# ── Imports ───────────────────────────────────────────────────────────────────

import os
import json
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient, models


# ── User Config Input ─────────────────────────────────────────────────────────

print("=" * 60)
print("  Pipeline Configuration")
print("=" * 60)
DATABRICKS_BASE_URL = input("  Databricks Base URL (serving-endpoints): ").strip()
LLM_MODEL           = input("  LLM model endpoint name (Claude)        : ").strip()
EMBED_MODEL         = input("  Embedding model endpoint name           : ").strip()
print("=" * 60 + "\n")

COLLECTION_NAME = "health_feedback_tag_index"
BATCH_SIZE      = 5
TOP_K           = 3
MAX_SYNONYMS    = 5


# ── Databricks OpenAI-Compatible Client ──────────────────────────────────────
#
# Single client for both LLM and embedding calls.
# Set token via: os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(...)

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

db_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TagSchema(BaseModel):
    tags: List[str] = Field(
        description=(
            "Array of 2 to 3 word descriptive phrases capturing explicit concepts, "
            "entities, or complaints in THIS comment only. "
            "Always expand abbreviations to full terms: "
            "'prior authorization' not 'prior auth', "
            "'explanation of benefits' not 'EOB', "
            "'out of pocket maximum' not 'OOP max', "
            "'procedure code' not 'CPT code'. "
            "Never use acronym-only tags. Max 4 tags."
        )
    )


# ── LLM Helpers ───────────────────────────────────────────────────────────────

def llm_call(system_prompt: str, user_content: str) -> str:
    response = db_client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ]
    )
    return response.choices[0].message.content


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences defensively before JSON parsing."""
    cleaned = raw.strip()
    # Strip opening fence variants
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    # Strip closing fence
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# ── Batch Tag Extraction ──────────────────────────────────────────────────────

BATCH_SYSTEM_PROMPT = """\
You are an expert healthcare operations analyst.
You will receive a numbered list of customer feedback comments.

For EACH comment independently (do not let one comment influence another), extract:
  tags: 2-4 descriptive 2-3 word phrases capturing the explicit complaints,
        entities, or issues mentioned in THAT comment only.

CRITICAL — abbreviation rules:
  - Always expand ALL abbreviations to full clinical terms.
  - Write "prior authorization" not "prior auth" or "PA"
  - Write "explanation of benefits" not "EOB"
  - Write "out of pocket maximum" not "OOP max"
  - Write "procedure code" not "CPT code"
  - Write "emergency room" not "ER"
  - Never create a tag that is only an acronym or abbreviation.

Good tag examples:
  "prior authorization denial", "long hold time", "billing code error",
  "portal upload failure", "explanation of benefits discrepancy",
  "specialist referral denied", "out of pocket maximum exceeded"

Return a JSON object with a single key "results" — an ordered array,
one object per input comment, each with a single key "tags" (array of strings).
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def extract_tags_batch(comments: List[str]) -> List[TagSchema]:
    numbered_input = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    raw    = llm_call(BATCH_SYSTEM_PROMPT, numbered_input)
    parsed = parse_json_response(raw)
    return [TagSchema(tags=item["tags"]) for item in parsed["results"]]


def extract_tags_in_batches(all_comments: List[str]) -> List[TagSchema]:
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE):
        chunk = all_comments[start : start + BATCH_SIZE]
        print(f"  Extracting tags — batch [{start+1}–{start+len(chunk)}] ...")
        all_results.extend(extract_tags_batch(chunk))
    return all_results


# ── Synonym Expansion ─────────────────────────────────────────────────────────

SYNONYM_SYSTEM_PROMPT = f"""\
You are a healthcare terminology specialist.
Given a search keyword or phrase related to health insurance,
return up to {MAX_SYNONYMS} synonyms, expansions, or related phrasings.

Rules:
  - Include both abbreviated and fully expanded clinical forms
  - Include colloquial variants a member might use
  - Include the original phrase as the first item
  - No duplicates
  - Keep each phrase short (2-4 words max)

Return a JSON object with a single key "synonyms" (array of strings).
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def expand_query_synonyms(keyword: str) -> List[str]:
    print(f"\n  [Synonyms] Expanding '{keyword}' ...")
    raw      = llm_call(SYNONYM_SYSTEM_PROMPT, f"Keyword: {keyword}")
    parsed   = parse_json_response(raw)
    synonyms = parsed.get("synonyms", [keyword])
    if keyword not in synonyms:
        synonyms = [keyword] + synonyms
    synonyms = synonyms[:MAX_SYNONYMS]
    print(f"  [Synonyms] → {synonyms}")
    return synonyms


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """
    Calls the Databricks embedding endpoint.
    Returns a plain Python list of floats.
    Qdrant requires a plain list — not a numpy array.
    """
    response = db_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


# ── Qdrant Init ───────────────────────────────────────────────────────────────
#
# FIX 1 — Embedding dimension resolved BEFORE collection creation.
#          A test embed call runs here so EMBED_DIM is always accurate
#          for the actual model endpoint in use.
#
# FIX 2 — Collection always recreated fresh at startup.
#          Stale in-memory collections from prior runs cause zero results
#          because the old collection has no points. Drop-and-recreate
#          guarantees a clean state on every run.
#
# FIX 3 — Payload index created AFTER collection but BEFORE any upserts.
#          Creating the index after upserts means existing points are
#          not indexed and MatchText returns nothing. Order matters.

print("Resolving embedding dimension ...")
_test_vector = embed_text("test")
EMBED_DIM    = len(_test_vector)
print(f"  Embedding dimension : {EMBED_DIM}")

print("Initialising Qdrant ...")
qdrant = QdrantClient(":memory:")

# Drop and recreate to guarantee clean state
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(COLLECTION_NAME)
    print(f"  Dropped existing collection '{COLLECTION_NAME}'")

qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(
        size=EMBED_DIM,
        distance=models.Distance.COSINE
    )
)

# Payload index created BEFORE upserts so all points get indexed
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="tag_text",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=30,
        lowercase=True
    )
)
print(f"  Collection '{COLLECTION_NAME}' created (dim={EMBED_DIM}, full-text index on tag_text)")


# ── Indexing ──────────────────────────────────────────────────────────────────

_point_id_counter = 0

def index_document(doc_id: int, text: str, tags: List[str]):
    """
    Exploded index — one Qdrant point per tag.
    Each point: dense embedding of the tag phrase + payload with
    doc_id, original text, this tag, and all tags for the document.
    """
    global _point_id_counter
    points = []

    for tag in tags:
        _point_id_counter += 1
        vector = embed_text(tag)

        # FIX 4 — explicit list cast guards against any SDK that returns
        # numpy arrays or other iterables from the embeddings endpoint
        if not isinstance(vector, list):
            vector = list(vector)

        points.append(
            models.PointStruct(
                id=_point_id_counter,
                vector=vector,
                payload={
                    "doc_id":        doc_id,
                    "original_text": text,
                    "tag_text":      tag,
                    "all_tags":      tags
                }
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Indexed doc {doc_id} — {len(tags)} tag points (IDs {_point_id_counter-len(tags)+1}–{_point_id_counter})")


# ── Diagnostics ───────────────────────────────────────────────────────────────

def run_diagnostics():
    """
    Prints collection state after ingest so you can verify points
    are actually present before search runs.
    Checks: point count, sample payload, sample vector non-zero.
    """
    info   = qdrant.get_collection(COLLECTION_NAME)
    count  = info.points_count
    print(f"\n  [Diag] Collection point count : {count}")

    if count == 0:
        print("  [Diag] WARNING — collection is empty, search will return nothing")
        return

    # Fetch one point to verify payload and vector shape
    sample = qdrant.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[1],
        with_payload=True,
        with_vectors=True
    )
    if sample:
        p = sample[0]
        vec_len    = len(p.vector) if p.vector else 0
        vec_nonzero = sum(1 for v in p.vector if v != 0.0) if p.vector else 0
        print(f"  [Diag] Sample point ID 1:")
        print(f"         tag_text  : {p.payload.get('tag_text')}")
        print(f"         doc_id    : {p.payload.get('doc_id')}")
        print(f"         vector dim: {vec_len}  non-zero: {vec_nonzero}")
        if vec_nonzero == 0:
            print("  [Diag] WARNING — vector is all zeros, embedding call may have failed")
    print()


# ── Query: Dual-Stream Search ─────────────────────────────────────────────────

def search_dual_stream(keyword: str, top_k: int = TOP_K) -> dict:
    """
    Stream A — Context search (dense embedding + synonym expansion)
        Embeds each synonym and runs query_points() against the
        tag vector index. Merges by doc_id, keeps best score.

    Stream B — Exact tag search (full-text token filter)
        query_points() with MatchText filter on tag_text field.
        Word-level: "auth" matches "prior authorization denial".
        No vector involved — pure payload filter pass.

    Both streams return separate ranked lists.
    Overlap (doc_id in both) flagged as highest confidence.
    """

    # ── Stream A ──────────────────────────────────────────────────────────────
    synonyms   = expand_query_synonyms(keyword)
    seen: dict = {}

    for syn in synonyms:
        vector = embed_text(syn)
        if not isinstance(vector, list):
            vector = list(vector)

        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True
        )

        print(f"  [Stream A] syn='{syn}' → {len(response.points)} hits")

        for hit in response.points:
            doc_id = hit.payload["doc_id"]
            if doc_id not in seen or hit.score > seen[doc_id]["score"]:
                seen[doc_id] = {
                    "doc_id":      doc_id,
                    "score":       hit.score,
                    "text":        hit.payload["original_text"],
                    "matched_tag": hit.payload["tag_text"],
                    "all_tags":    hit.payload["all_tags"],
                    "matched_syn": syn
                }

    context_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # ── Stream B ──────────────────────────────────────────────────────────────
    tag_response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tag_text",
                    match=models.MatchText(text=keyword)
                )
            ]
        ),
        limit=top_k,
        with_payload=True
    )

    print(f"  [Stream B] keyword='{keyword}' → {len(tag_response.points)} hits")

    tag_seen: dict = {}
    for hit in tag_response.points:
        doc_id = hit.payload["doc_id"]
        if doc_id not in tag_seen:
            tag_seen[doc_id] = {
                "doc_id":      doc_id,
                "text":        hit.payload["original_text"],
                "matched_tag": hit.payload["tag_text"],
                "all_tags":    hit.payload["all_tags"]
            }

    tag_results = list(tag_seen.values())[:top_k]

    # ── Overlap ───────────────────────────────────────────────────────────────
    context_ids = {r["doc_id"] for r in context_results}
    tag_ids     = {r["doc_id"] for r in tag_results}
    overlap_ids = context_ids & tag_ids

    return {
        "context_results": context_results,
        "tag_results":     tag_results,
        "overlap_ids":     overlap_ids
    }


# ── Result Rendering ──────────────────────────────────────────────────────────

def render_dual_stream_results(keyword: str, results: dict):
    overlap_ids = results["overlap_ids"]
    sep         = "─" * 62

    print(f"\n{'=' * 62}")
    print(f"  Keyword : '{keyword}'")
    print(f"{'=' * 62}")

    print(f"\n  Stream A — Context Search (Embedding + Synonyms)")
    print(f"  {sep}")
    if not results["context_results"]:
        print("  No results — check diagnostics above for root cause.")
    for rank, r in enumerate(results["context_results"], start=1):
        flag = "  ◀ OVERLAP" if r["doc_id"] in overlap_ids else ""
        print(f"\n  [A{rank}]  Score: {r['score']:.4f}  Doc: {r['doc_id']}{flag}")
        print(f"  Matched tag : '{r['matched_tag']}'  (via synonym: '{r['matched_syn']}')")
        print(f"  All tags    : {r['all_tags']}")
        print(f"  Text        : {r['text'][:100]}...")

    print(f"\n  Stream B — Exact Tag Search (Full-Text Token Match)")
    print(f"  {sep}")
    if not results["tag_results"]:
        print("  No results — keyword tokens not found in any stored tag.")
    for rank, r in enumerate(results["tag_results"], start=1):
        flag = "  ◀ OVERLAP" if r["doc_id"] in overlap_ids else ""
        print(f"\n  [B{rank}]  Doc: {r['doc_id']}{flag}")
        print(f"  Matched tag : '{r['matched_tag']}'")
        print(f"  All tags    : {r['all_tags']}")
        print(f"  Text        : {r['text'][:100]}...")

    print(f"\n  Overlap Summary")
    print(f"  {sep}")
    if overlap_ids:
        print(f"  Doc IDs in both streams : {sorted(overlap_ids)}")
        print(f"  Highest confidence — contextually relevant AND lexically matched.")
    else:
        print(f"  No overlap — streams returned disjoint results.")
    print(f"{'=' * 62}\n")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    raw_feedbacks = [
        "I was on hold for two full hours, and when the agent finally picked up, "
        "they told me my prior authorization for the knee MRI was rejected.",
        "The portal gave me a network error when I tried uploading my claim documentation. "
        "Now I am missing the filing deadline!",
        "The medical treatment was fine, but the billing department sent me a huge invoice "
        "because they typed the coding wrong.",
        "My specialist referral was denied without any explanation and I cannot reach anyone "
        "in the approvals department.",
        "I received two EOBs for the same procedure with different amounts and now I do not "
        "know which one to pay.",
    ]

    # ── Step 1: Batch Tag Extraction + Indexing ───────────────────────────────
    print("=== Step 1: Batch Tag Extraction (Claude, batch_size=5) ===\n")
    extracted_tags_list = extract_tags_in_batches(raw_feedbacks)

    print("\n=== Indexing Documents ===\n")
    for idx, (comment, tag_data) in enumerate(
        zip(raw_feedbacks, extracted_tags_list), start=1
    ):
        print(f"  Doc {idx} : {comment[:75]}...")
        print(f"  Tags    : {tag_data.tags}")
        index_document(doc_id=idx, text=comment, tags=tag_data.tags)

    # ── Diagnostics — verify collection state before searching ───────────────
    print("\n=== Collection Diagnostics ===")
    run_diagnostics()

    # ── Step 2: Dual-Stream Search ────────────────────────────────────────────
    print("=== Step 2: Dual-Stream Tag Search ===")

    for keyword in ["prior auth", "billing", "portal error"]:
        results = search_dual_stream(keyword=keyword, top_k=TOP_K)
        render_dual_stream_results(keyword=keyword, results=results)
