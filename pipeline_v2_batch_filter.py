# =============================================================================
# pipeline_v1_batch_filter.py
#
# ARCHITECTURE: Batch LLM Tag Extraction + Dual-Stream Tag Search
#
# Ingest phase:
#   - Claude extracts tags only (intents removed) in batches of 5
#   - Tags stored as payload; full-text index built on tags field
#     for word-level token matching (not brittle exact string equality)
#   - Original comment text encoded with FastEmbed SPLADE
#
# Query phase — two separate ranked streams returned:
#   Stream A | Context search
#             SPLADE encodes the raw keyword and scores all documents
#             by sparse dot product. Surfacing is vocabulary-aware —
#             "prior auth" matches "authorization rejected" via learned
#             token weights. No filter applied; full index is scored.
#
#   Stream B | Exact tag search
#             Qdrant full-text filter on the tags field. Word-level
#             tokenization means "auth" matches "prior auth denial" and
#             "authorization issue" — better than MatchValue string
#             equality but still anchored to literal token presence.
#
#   Docs appearing in both streams are flagged as overlap — these are
#   the highest-confidence results since they satisfy both context
#   relevance and tag-level lexical evidence.
#
# FastEmbed note:
#   Downloads from its own CDN (not HuggingFace Hub). Compatible with
#   Databricks environments where huggingface.co is blocked.
#   Model: prithvida/Splade_PP_en_v1
# =============================================================================

# %pip install "qdrant-client[fastembed]" langchain-core langchain-openai pydantic
# dbutils.library.restartPython()


# ── Imports ───────────────────────────────────────────────────────────────────

import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models


# ── Configuration ─────────────────────────────────────────────────────────────

DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get(scope="<your-scope>", key="databricks-token")
MODEL_ENDPOINT   = "databricks-claude-sonnet"
COLLECTION_NAME  = "structured_health_feedback"
BATCH_SIZE       = 5


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class TagSchema(BaseModel):
    """
    Tags-only extraction schema. Intents removed.
    Tags are 2-3 word phrases describing the explicit complaint or entity
    in the comment — used for both SPLADE payload context and full-text
    token matching at query time.
    """
    tags: List[str] = Field(
        description=(
            "Array of 2 to 3 word highly descriptive phrases capturing explicit concepts, "
            "entities, or complaints in THIS comment only "
            "(e.g., 'mri denial', 'long hold time', 'billing code error'). Max 4 tags."
        )
    )


# ── LLM Init ──────────────────────────────────────────────────────────────────

def build_llm() -> ChatOpenAI:
    """Databricks OpenAI-compatible endpoint for Claude."""
    return ChatOpenAI(
        model=MODEL_ENDPOINT,
        temperature=0.0,
        openai_api_key=DATABRICKS_TOKEN,
        openai_api_base=f"{DATABRICKS_HOST}/serving-endpoints",
    )


# ── Batch Tag Extraction ──────────────────────────────────────────────────────

BATCH_SYSTEM_PROMPT = """\
You are an expert healthcare operations analyst.
You will receive a numbered list of customer feedback comments.
For EACH comment independently (do not let one comment influence another),
extract tags: 2-4 descriptive 2-3 word phrases capturing the explicit
complaints, entities, or issues mentioned in THAT comment only.

Good tag examples: 'mri denial', 'long hold time', 'billing code error',
'portal upload failure', 'eob discrepancy', 'specialist referral denied'.

Return a JSON object with a single key "results" containing an ordered array,
one object per input comment, each with a single key "tags" (array of strings).
Return ONLY valid JSON -- no preamble, no markdown fences.
"""

def extract_tags_batch(comments: List[str]) -> List[TagSchema]:
    """
    Sends up to BATCH_SIZE comments in a single Claude call.
    Returns an ordered list of TagSchema, one per comment.

    Numbered I/O format enforces per-comment isolation — the primary
    mitigation for tag contamination across comments in the same batch.
    """
    llm = build_llm()

    numbered_input = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))

    prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_SYSTEM_PROMPT),
        ("human", "{numbered_comments}")
    ])

    chain    = prompt | llm
    response = chain.invoke({"numbered_comments": numbered_input})

    raw    = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    parsed = json.loads(raw)

    return [TagSchema(tags=item["tags"]) for item in parsed["results"]]


def extract_tags_in_batches(all_comments: List[str]) -> List[TagSchema]:
    """Chunks the full comment list into BATCH_SIZE groups and extracts tags."""
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE):
        chunk = all_comments[start : start + BATCH_SIZE]
        print(f"  Processing batch [{start+1}-{start+len(chunk)}] ...")
        all_results.extend(extract_tags_batch(chunk))
    return all_results


# ── FastEmbed SPLADE Init ─────────────────────────────────────────────────────
#
# First run downloads weights to ~/.cache/fastembed/ from Google CDN.
# Subsequent runs are fully local — no network call needed.
# Pre-warm this cell on a new cluster before running the main pipeline.

print("Initialising FastEmbed SPLADE model (first run will download weights) ...")
splade_model = SparseTextEmbedding(model_name="prithvida/Splade_PP_en_v1")
print("SPLADE model ready.")


# ── Qdrant Init ───────────────────────────────────────────────────────────────

print("Initialising Qdrant storage engine ...")
client = QdrantClient(":memory:")

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={},
        sparse_vectors_config={
            "splade_vector": models.SparseVectorParams()
        }
    )

    # Full-text index on the tags field.
    # WORD tokenizer splits tags into individual tokens so that a query
    # keyword "auth" matches stored tags "prior auth denial",
    # "authorization issue" etc. — better than MatchValue string equality
    # while remaining anchored to literal token presence.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tags",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True        # normalise case so "MRI" matches "mri"
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created with full-text tag index.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")


# ── SPLADE Encoding ───────────────────────────────────────────────────────────

def compute_splade_vector(text: str) -> dict:
    """
    Encodes text via FastEmbed SPLADE.
    Returns {indices, values} as plain Python lists for Qdrant.
    .tolist() is required — FastEmbed returns numpy arrays.
    """
    embedding = list(splade_model.embed([text]))[0]
    return {
        "indices": embedding.indices.tolist(),
        "values":  embedding.values.tolist()
    }


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_document(doc_id: int, text: str, tags: list):
    """
    Encodes the comment text with SPLADE and upserts it into Qdrant.
    Payload stores original_text and tags.
    Tags are indexed by the full-text payload index created at collection
    init — no extra step needed here.
    """
    sparse_data = compute_splade_vector(text)
    payload = {
        "original_text": text,
        "tags":           tags
    }
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=doc_id,
                vector={
                    "splade_vector": models.SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"]
                    )
                },
                payload=payload
            )
        ]
    )


# ── Query: Dual-Stream Search ─────────────────────────────────────────────────

def search_dual_stream(
    keyword: str,
    top_k:   int = 3
) -> dict:
    """
    Runs two independent searches against the same keyword and returns
    them as separate ranked lists so results can be inspected side by side.

    Stream A — Context search (SPLADE, no filter)
        Encodes the keyword with SPLADE and scores ALL documents by
        sparse dot product. Context-aware: "prior auth" matches
        "authorization rejected" via learned token weight overlap.
        Nothing is pre-filtered out.

    Stream B — Exact tag search (full-text filter, no vector scoring)
        Uses Qdrant's full-text payload filter on the tags field.
        Word-level tokenisation: keyword tokens must be present in at
        least one stored tag string. Returns documents ranked by
        Qdrant's internal text relevance, not SPLADE score.
        "auth" matches "prior auth denial" and "authorization issue".

    Overlap — docs appearing in both streams
        Identified by doc ID intersection. These are the
        highest-confidence results: contextually relevant AND
        lexically matched in tags.

    Returns a dict with keys:
        context_results  : list of hits from Stream A, scored by SPLADE
        tag_results      : list of hits from Stream B, scored by text match
        overlap_ids      : set of doc IDs present in both streams
    """
    # ── Stream A: context search ──────────────────────────────────────────────
    query_data = compute_splade_vector(keyword)

    sparse_query_vector = models.NamedSparseVector(
        name="splade_vector",
        vector=models.SparseVector(
            indices=query_data["indices"],
            values=query_data["values"]
        )
    )

    context_hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=sparse_query_vector,
        query_filter=None,     # no filter — full index scored
        limit=top_k
    )

    context_results = [
        {
            "doc_id": hit.id,
            "score":  hit.score,
            "text":   hit.payload["original_text"],
            "tags":   hit.payload["tags"]
        }
        for hit in context_hits
    ]

    # ── Stream B: exact tag search ────────────────────────────────────────────
    # MatchText on the tags field uses the full-text index.
    # The keyword is tokenised and matched against stored tag tokens.
    tag_hits = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchText(text=keyword)   # full-text token match
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
        with_vectors=False
    )[0]   # scroll() returns (records, next_page_offset)

    tag_results = [
        {
            "doc_id": hit.id,
            "text":   hit.payload["original_text"],
            "tags":   hit.payload["tags"]
        }
        for hit in tag_hits
    ]

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

    print(f"\n{'='*65}")
    print(f"  Keyword: '{keyword}'")
    print(f"{'='*65}")

    # Stream A
    print(f"\n── Stream A: Context Search (SPLADE) ──────────────────────")
    if not results["context_results"]:
        print("  No results.")
    for rank, r in enumerate(results["context_results"], start=1):
        flag = "  *** OVERLAP ***" if r["doc_id"] in overlap_ids else ""
        print(f"\n  [Rank {rank}]  Score: {r['score']:.4f}  Doc ID: {r['doc_id']}{flag}")
        print(f"  Text : {r['text'][:100]}...")
        print(f"  Tags : {r['tags']}")

    # Stream B
    print(f"\n── Stream B: Exact Tag Search (Full-Text) ─────────────────")
    if not results["tag_results"]:
        print("  No results — keyword tokens not found in any stored tag.")
    for rank, r in enumerate(results["tag_results"], start=1):
        flag = "  *** OVERLAP ***" if r["doc_id"] in overlap_ids else ""
        print(f"\n  [Rank {rank}]  Doc ID: {r['doc_id']}{flag}")
        print(f"  Text : {r['text'][:100]}...")
        print(f"  Tags : {r['tags']}")

    # Overlap summary
    print(f"\n── Overlap Summary ────────────────────────────────────────")
    if overlap_ids:
        print(f"  Doc IDs in both streams: {sorted(overlap_ids)}")
        print(f"  These docs are contextually relevant AND lexically matched in tags.")
    else:
        print(f"  No overlap — context and tag streams returned disjoint results.")
    print(f"{'='*65}\n")


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
        print(f"Doc {idx}:")
        print(f"  Comment : {comment[:80]}...")
        print(f"  Tags    : {tag_data.tags}\n")
        index_document(doc_id=idx, text=comment, tags=tag_data.tags)

    # ── Step 2: Dual-Stream Search ────────────────────────────────────────────
    print("=== Step 2: Dual-Stream Tag Search ===")

    # Run multiple keywords to show both streams behaving differently
    for keyword in ["prior auth", "billing", "portal error"]:
        results = search_dual_stream(keyword=keyword, top_k=3)
        render_dual_stream_results(keyword=keyword, results=results)
