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
#   - Full-text index also built on tags for exact token stream
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
#   DATABRICKS_BASE_URL and DATABRICKS_MODEL read from user input at startup.
# =============================================================================

# %pip install openai qdrant-client pydantic
# dbutils.library.restartPython()


# ── Imports ───────────────────────────────────────────────────────────────────

import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client import models


# ── User Config Input ─────────────────────────────────────────────────────────
#
# Base URL  : your Databricks workspace instance URL
#             e.g. https://<workspace>.azuredatabricks.net/serving-endpoints
# LLM model : your Claude serving endpoint name
#             e.g. databricks-claude-sonnet
# Embed model: your embedding model serving endpoint name
#             e.g. databricks-bge-large-en

print("=" * 60)
print("  Pipeline Configuration")
print("=" * 60)
DATABRICKS_BASE_URL = input("  Databricks Base URL (serving-endpoints): ").strip()
LLM_MODEL           = input("  LLM model endpoint name (Claude)        : ").strip()
EMBED_MODEL         = input("  Embedding model endpoint name           : ").strip()
print("=" * 60 + "\n")

COLLECTION_NAME     = "health_feedback_tag_index"
BATCH_SIZE          = 5
TOP_K               = 3
MAX_SYNONYMS        = 5    # cap synonym expansion to limit parallel searches


# ── OpenAI-Compatible Databricks Client ───────────────────────────────────────
#
# Single client instance used for both LLM calls (Claude) and
# embedding calls. Token read from environment — set via:
#   os.environ["DATABRICKS_TOKEN"] = dbutils.secrets.get(...)
# or export DATABRICKS_TOKEN=<pat> before running.

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TagSchema(BaseModel):
    """
    Tags-only extraction schema.
    All abbreviations must be expanded to full clinical terms.
    Tags are 2-3 word phrases describing the explicit complaint or entity.
    """
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


class SynonymSchema(BaseModel):
    """Synonym and expansion list for a query keyword."""
    synonyms: List[str] = Field(
        description=(
            f"List of up to {MAX_SYNONYMS} synonyms, expansions, or related phrasings "
            "for the input term. Include both abbreviated and expanded forms, "
            "clinical and colloquial variants. No duplicates."
        )
    )


# ── LLM Helpers ───────────────────────────────────────────────────────────────

def llm_call(system_prompt: str, user_content: str) -> str:
    """
    Single LLM call via Databricks OpenAI-compatible client.
    Returns the raw string content of the first choice message.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content}
        ]
    )
    return response.choices[0].message.content


def parse_json_response(raw: str) -> dict:
    """Strips markdown fences defensively before JSON parsing."""
    cleaned = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(cleaned)


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
    """
    Sends up to BATCH_SIZE comments in one Claude call.
    Numbered I/O enforces per-comment isolation — primary
    mitigation for tag contamination across batch items.
    """
    numbered_input = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    raw            = llm_call(BATCH_SYSTEM_PROMPT, numbered_input)
    parsed         = parse_json_response(raw)
    return [TagSchema(tags=item["tags"]) for item in parsed["results"]]


def extract_tags_in_batches(all_comments: List[str]) -> List[TagSchema]:
    """Chunks the full comment list into BATCH_SIZE groups."""
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
    """
    Calls Claude to generate synonyms and expansions for the search keyword.
    Always includes the original keyword as the first entry.
    """
    print(f"\n  [Synonyms] Expanding '{keyword}' ...")
    raw     = llm_call(SYNONYM_SYSTEM_PROMPT, f"Keyword: {keyword}")
    parsed  = parse_json_response(raw)
    synonyms = parsed.get("synonyms", [keyword])
    # Guarantee original keyword is present even if Claude omits it
    if keyword not in synonyms:
        synonyms = [keyword] + synonyms
    print(f"  [Synonyms] → {synonyms}")
    return synonyms[:MAX_SYNONYMS]


# ── Embedding Helper ──────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """
    Calls the Databricks embedding model endpoint via the same
    OpenAI-compatible client used for LLM calls.
    Returns the embedding vector as a plain Python list of floats.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


# ── Qdrant Init ───────────────────────────────────────────────────────────────
#
# Collection stores ONE POINT PER TAG (exploded index pattern).
# Each tag gets its own dense embedding vector.
# The doc_id in payload groups points back to the original comment.
# This ensures search is always short-phrase vs short-phrase,
# avoiding the short-text degradation of embedding full comments.
#
# Embedding dimension is resolved dynamically from a test call
# so this file works with any embedding model endpoint.

print("Resolving embedding dimension from model endpoint ...")
_test_vector = embed_text("test")
EMBED_DIM    = len(_test_vector)
print(f"Embedding dimension: {EMBED_DIM}")

print("Initialising Qdrant ...")
qdrant = QdrantClient(":memory:")

if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE
        )
    )

    # Full-text index for Stream B (exact token matching on tag strings)
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
    print(f"Collection '{COLLECTION_NAME}' created with full-text tag index.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")


# ── Indexing ──────────────────────────────────────────────────────────────────

# Global point ID counter — incremented for each tag point inserted
_point_id_counter = 0

def index_document(doc_id: int, text: str, tags: List[str]):
    """
    Exploded index: inserts one Qdrant point per tag.
    Each point stores:
      - Dense embedding of the tag phrase
      - Payload: doc_id, original comment text, this tag, all tags for the doc

    doc_id    : original document identifier (groups tag points back to comment)
    text      : original comment text (stored for result display)
    tags      : list of tag strings generated by Claude for this comment
    """
    global _point_id_counter
    points = []

    for tag in tags:
        _point_id_counter += 1
        vector = embed_text(tag)
        points.append(
            models.PointStruct(
                id=_point_id_counter,
                vector=vector,
                payload={
                    "doc_id":        doc_id,
                    "original_text": text,
                    "tag_text":      tag,      # single tag — used for full-text index
                    "all_tags":      tags      # all tags for this doc — shown in results
                }
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Query: Dual-Stream Search ─────────────────────────────────────────────────

def search_dual_stream(keyword: str, top_k: int = TOP_K) -> dict:
    """
    Runs two independent searches and returns them as separate ranked lists.

    Stream A — Context search (dense embedding + synonym expansion)
        1. Claude expands keyword to up to MAX_SYNONYMS synonyms
        2. Each synonym is embedded and searched independently
        3. All hits merged by doc_id, best tag score per doc kept
        4. Final list sorted by score descending
        Context-aware: "prior auth" expands to "prior authorization",
        "preauthorization" etc. and matches semantically similar tags.

    Stream B — Exact tag search (full-text token filter)
        Qdrant MatchText on tag_text field. Word-level tokenisation:
        keyword tokens must be present in a stored tag string.
        "auth" matches "prior authorization denial" token-by-token.
        No embedding involved — anchored to literal token presence.

    Overlap — doc_ids present in both streams are flagged.
    These are highest-confidence: semantically relevant AND
    lexically matched.

    Returns dict with keys:
        context_results : list sorted by best embedding score per doc
        tag_results     : list from full-text filter stream
        overlap_ids     : set of doc_ids in both streams
    """

    # ── Stream A: synonym expansion + dense embedding search ─────────────────
    synonyms  = expand_query_synonyms(keyword)
    seen: dict = {}

    for syn in synonyms:
        vector  = embed_text(syn)
        hits    = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k
        )
        for hit in hits:
            doc_id = hit.payload["doc_id"]
            # Keep highest score across all synonym searches for this doc
            if doc_id not in seen or hit.score > seen[doc_id]["score"]:
                seen[doc_id] = {
                    "doc_id":       doc_id,
                    "score":        hit.score,
                    "text":         hit.payload["original_text"],
                    "matched_tag":  hit.payload["tag_text"],
                    "all_tags":     hit.payload["all_tags"],
                    "matched_syn":  syn
                }

    context_results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    # ── Stream B: full-text token match on tag_text ───────────────────────────
    tag_hits = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tag_text",
                    match=models.MatchText(text=keyword)
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )[0]

    # Deduplicate by doc_id — keep first occurrence per doc
    tag_seen: dict = {}
    for hit in tag_hits:
        doc_id = hit.payload["doc_id"]
        if doc_id not in tag_seen:
            tag_seen[doc_id] = {
                "doc_id":      doc_id,
                "text":        hit.payload["original_text"],
                "matched_tag": hit.payload["tag_text"],
                "all_tags":    hit.payload["all_tags"]
            }

    tag_results = list(tag_seen.values())[:top_k]

    # ── Overlap detection ─────────────────────────────────────────────────────
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
    """Prints both streams side by side with overlap flagged."""
    overlap_ids = results["overlap_ids"]
    sep         = "─" * 62

    print(f"\n{'=' * 62}")
    print(f"  Keyword : '{keyword}'")
    print(f"{'=' * 62}")

    # Stream A
    print(f"\n  Stream A — Context Search (Embedding + Synonyms)")
    print(f"  {sep}")
    if not results["context_results"]:
        print("  No results.")
    for rank, r in enumerate(results["context_results"], start=1):
        flag = "  ◀ OVERLAP" if r["doc_id"] in overlap_ids else ""
        print(f"\n  [A{rank}]  Score: {r['score']:.4f}  Doc: {r['doc_id']}{flag}")
        print(f"  Matched tag : '{r['matched_tag']}'  (via synonym: '{r['matched_syn']}')")
        print(f"  All tags    : {r['all_tags']}")
        print(f"  Text        : {r['text'][:100]}...")

    # Stream B
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

    # Overlap summary
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

    for idx, (comment, tag_data) in enumerate(
        zip(raw_feedbacks, extracted_tags_list), start=1
    ):
        print(f"  Doc {idx} : {comment[:75]}...")
        print(f"  Tags    : {tag_data.tags}\n")
        index_document(doc_id=idx, text=comment, tags=tag_data.tags)

    print(f"  Indexed {_point_id_counter} tag points across {len(raw_feedbacks)} documents.\n")

    # ── Step 2: Dual-Stream Search ────────────────────────────────────────────
    print("=== Step 2: Dual-Stream Tag Search ===")

    for keyword in ["prior auth", "billing", "portal error"]:
        results = search_dual_stream(keyword=keyword, top_k=TOP_K)
        render_dual_stream_results(keyword=keyword, results=results)
