# =============================================================================
# pipeline_v1_batch_filter.py
#
# ARCHITECTURE: Batch LLM Extraction + Parallel Metadata-Filtered Sparse Search
#
# Key changes from original:
#   1. Claude via Databricks Model Serving (OpenAI-compatible endpoint)
#   2. LLM extraction batched at 5 comments per call
#   3. SPLADE encoding via FastEmbed (replaces HuggingFace + torch)
#   4. Query phase uses a single keyword/phrase with parallel
#      SPLADE vector search + intent AND tag metadata pre-filters
#
# FastEmbed note:
#   fastembed bundles model weights and downloads from its own CDN
#   (not HuggingFace Hub), making it compatible with Databricks
#   environments where huggingface.co is blocked.
#   Model used: prithvida/Splade_PP_en_v1 (production SPLADE variant)
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

# Replace with your actual Databricks workspace URL and secret scope/key
DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get(scope="<your-scope>", key="databricks-token")
MODEL_ENDPOINT   = "databricks-claude-sonnet"   # your serving endpoint name
COLLECTION_NAME  = "structured_health_feedback"
BATCH_SIZE       = 5


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class StructuredFeedbackSchema(BaseModel):
    """Schema for a single comment's extracted metadata."""
    intents: List[str] = Field(
        description=(
            "Array of applicable canonical intents. Must choose from: "
            "'Billing Dispute', 'Coverage Inquiry', 'Customer Service Issue', 'Technical Error'."
        )
    )
    tags: List[str] = Field(
        description=(
            "Array of 2 to 3 word highly descriptive phrases capturing explicit concepts, "
            "entities, or complaints (e.g., 'mri denial', 'long hold time'). Max 4 tags."
        )
    )


# ── LLM Init ──────────────────────────────────────────────────────────────────

def build_llm() -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI instance pointed at the Databricks
    OpenAI-compatible model serving endpoint for Claude.
    """
    return ChatOpenAI(
        model=MODEL_ENDPOINT,
        temperature=0.0,
        openai_api_key=DATABRICKS_TOKEN,
        openai_api_base=f"{DATABRICKS_HOST}/serving-endpoints",
    )


# ── Batch Metadata Extraction ─────────────────────────────────────────────────

BATCH_SYSTEM_PROMPT = """\
You are an expert healthcare operations analyst.
You will receive a numbered list of customer feedback comments.
For EACH comment independently (do not let one comment influence another),
extract:
  - intents: one or more of 'Billing Dispute', 'Coverage Inquiry',
             'Customer Service Issue', 'Technical Error'
  - tags: 2-4 descriptive 2-3 word phrases capturing explicit complaints
          or entities mentioned in THAT comment only.

Return a JSON object with a single key "results" containing an ordered array,
one object per input comment, each with keys "intents" and "tags".
Return ONLY valid JSON -- no preamble, no markdown fences.
"""

def extract_metadata_batch(comments: List[str]) -> List[StructuredFeedbackSchema]:
    """
    Sends up to BATCH_SIZE comments in a single Claude call.
    Returns an ordered list of StructuredFeedbackSchema, one per comment.

    Numbered I/O format enforces strict per-comment isolation and gives
    the model a clear mapping between input position and output array index,
    which is the primary mitigation for tag contamination in batch calls.
    """
    llm = build_llm()

    numbered_input = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))

    prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_SYSTEM_PROMPT),
        ("human", "{numbered_comments}")
    ])

    chain    = prompt | llm
    response = chain.invoke({"numbered_comments": numbered_input})

    # Strip markdown fences defensively before parsing
    raw    = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    parsed = json.loads(raw)

    return [
        StructuredFeedbackSchema(intents=item["intents"], tags=item["tags"])
        for item in parsed["results"]
    ]


def extract_metadata_in_batches(all_comments: List[str]) -> List[StructuredFeedbackSchema]:
    """Chunks the full comment list into BATCH_SIZE groups and extracts each."""
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE):
        chunk = all_comments[start : start + BATCH_SIZE]
        print(f"  Processing batch [{start+1}-{start+len(chunk)}] ...")
        all_results.extend(extract_metadata_batch(chunk))
    return all_results


# ── FastEmbed SPLADE Init ─────────────────────────────────────────────────────
#
# FastEmbed downloads model weights once on first use from its own CDN
# (storage.googleapis.com), not from HuggingFace Hub. Subsequent runs
# use the locally cached weights — no network call needed after first run.
#
# Cache location on Databricks: ~/.cache/fastembed/
# To pre-warm on a new cluster, run this cell once before the main pipeline.

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
    print(f"Collection '{COLLECTION_NAME}' created.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")


# ── SPLADE Encoding ───────────────────────────────────────────────────────────

def compute_splade_vector(text: str) -> dict:
    """
    Encodes a single text string using FastEmbed SPLADE.
    Returns a dict with 'indices' and 'values' lists ready for Qdrant.

    FastEmbed's embed() is a generator — we wrap in list() and take
    index [0] to extract the single SparseEmbedding object.
    .indices and .values are numpy arrays, so .tolist() is required
    before passing to Qdrant (which expects plain Python lists).
    """
    embedding = list(splade_model.embed([text]))[0]
    return {
        "indices": embedding.indices.tolist(),
        "values":  embedding.values.tolist()
    }


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_document(doc_id: int, text: str, intents: list, tags: list):
    """Encodes a document with SPLADE and upserts it with its metadata payload."""
    sparse_data = compute_splade_vector(text)
    payload = {
        "original_text": text,
        "intents":        intents,
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


# ── Query: Parallel Intent + Tag Filter ──────────────────────────────────────

def search_parallel_filters(
    keyword:       str,
    target_intent: Optional[str] = None,
    target_tag:    Optional[str] = None,
    top_k:         int = 3
) -> list:
    """
    Encodes the keyword with SPLADE then runs two Qdrant searches in parallel:
      - Stream A: sparse search pre-filtered by intent
      - Stream B: sparse search pre-filtered by tag

    Results are merged by doc ID. If a document surfaces in both streams
    it naturally ranks higher since its best score is preserved and it
    carries both stream labels. Final list is sorted by score descending.

    Both filters are optional — if neither is provided the function falls
    back to a single unfiltered search.

    Args:
        keyword:       Single word or short phrase from the user.
        target_intent: Canonical intent string to pre-filter on (optional).
        target_tag:    Tag phrase to pre-filter on (optional).
        top_k:         Number of results to return after merge.
    """
    query_data = compute_splade_vector(keyword)

    sparse_query_vector = models.NamedSparseVector(
        name="splade_vector",
        vector=models.SparseVector(
            indices=query_data["indices"],
            values=query_data["values"]
        )
    )

    seen    = {}   # doc_id -> best hit dict
    streams = []

    # Stream A — intent pre-filter
    if target_intent:
        streams.append((
            "intent",
            models.Filter(must=[
                models.FieldCondition(
                    key="intents",
                    match=models.MatchValue(value=target_intent)
                )
            ])
        ))

    # Stream B — tag pre-filter
    if target_tag:
        streams.append((
            "tag",
            models.Filter(must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchValue(value=target_tag)
                )
            ])
        ))

    # Fallback: no filters provided
    if not streams:
        streams.append(("unfiltered", None))

    # Execute streams and merge by doc ID
    for stream_label, search_filter in streams:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=sparse_query_vector,
            query_filter=search_filter,
            limit=top_k
        )
        for hit in results:
            doc_id = hit.id
            if doc_id not in seen or hit.score > seen[doc_id]["score"]:
                seen[doc_id] = {
                    "score":   hit.score,
                    "text":    hit.payload["original_text"],
                    "intents": hit.payload["intents"],
                    "tags":    hit.payload["tags"],
                    "streams": set()
                }
            seen[doc_id]["streams"].add(stream_label)

    merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


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

    # ── Step 1: Batch LLM Extraction + Indexing ───────────────────────────────
    print("=== Step 1: Batch Metadata Extraction (Claude, batch_size=5) ===")
    extracted_metadata_list = extract_metadata_in_batches(raw_feedbacks)

    for idx, (comment, metadata) in enumerate(
        zip(raw_feedbacks, extracted_metadata_list), start=1
    ):
        print(f"\nDoc {idx}:")
        print(f"  Comment : {comment[:80]}...")
        print(f"  Intents : {metadata.intents}")
        print(f"  Tags    : {metadata.tags}")
        index_document(
            doc_id=idx,
            text=comment,
            intents=metadata.intents,
            tags=metadata.tags
        )

    # ── Step 2: Parallel Filtered Search ─────────────────────────────────────
    print("\n=== Step 2: Parallel Intent + Tag Filtered Sparse Search ===")

    keyword       = "prior auth"
    intent_filter = "Coverage Inquiry"
    tag_filter    = "prior auth denial"   # should match a tag generated at ingest

    results = search_parallel_filters(
        keyword=keyword,
        target_intent=intent_filter,
        target_tag=tag_filter,
        top_k=3
    )

    print(f"\nKeyword : '{keyword}'")
    print(f"Filters : intent='{intent_filter}' | tag='{tag_filter}'\n")

    for rank, match in enumerate(results, start=1):
        print(f"[Rank {rank}]  Score: {match['score']:.4f}  |  Streams: {match['streams']}")
        print(f"  Text    : {match['text'][:100]}...")
        print(f"  Intents : {match['intents']}")
        print(f"  Tags    : {match['tags']}\n")
