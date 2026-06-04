# =============================================================================
# pipeline_v1_batch_filter.py
#
# ARCHITECTURE: Batch LLM Extraction + Parallel Metadata-Filtered Sparse Search
#
# Key changes from original:
#   1. Claude via Databricks Model Serving (OpenAI-compatible endpoint)
#   2. LLM extraction batched at 5 comments per call
#   3. Query phase uses a single keyword/phrase with parallel
#      SPLADE vector search + intent AND tag metadata pre-filters
# =============================================================================

# %pip install torch transformers qdrant-client langchain-core langchain-openai pydantic
# dbutils.library.restartPython()


# ── Imports ──────────────────────────────────────────────────────────────────

import json
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client import models


# ── Databricks / Claude Configuration ────────────────────────────────────────

# Replace with your actual Databricks workspace URL and model serving endpoint name
DATABRICKS_HOST    = "https://<your-workspace>.azuredatabricks.net"
DATABRICKS_TOKEN   = dbutils.secrets.get(scope="<your-scope>", key="databricks-token")
MODEL_ENDPOINT     = "databricks-claude-sonnet"   # your serving endpoint name
COLLECTION_NAME    = "structured_health_feedback"
BATCH_SIZE         = 5


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


class BatchedFeedbackSchema(BaseModel):
    """Wrapper schema for batch extraction — one entry per comment."""
    results: List[StructuredFeedbackSchema] = Field(
        description="Ordered list of extracted metadata, one per input comment."
    )


# ── LLM Initialisation ────────────────────────────────────────────────────────

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


# ── Batch Extraction ──────────────────────────────────────────────────────────

BATCH_SYSTEM_PROMPT = """\
You are an expert healthcare operations analyst.
You will receive a numbered list of customer feedback comments.
For EACH comment independently (do not let one comment influence another),
extract:
  - intents: one or more of 'Billing Dispute', 'Coverage Inquiry',
             'Customer Service Issue', 'Technical Error'
  - tags: 2–4 descriptive 2–3 word phrases capturing explicit complaints
          or entities mentioned in THAT comment only.

Return a JSON object with a single key "results" containing an ordered array,
one object per input comment, each with keys "intents" and "tags".
Return ONLY valid JSON — no preamble, no markdown fences.
"""

def extract_metadata_batch(comments: List[str]) -> List[StructuredFeedbackSchema]:
    """
    Sends up to BATCH_SIZE comments in a single Claude call.
    Returns an ordered list of StructuredFeedbackSchema, one per comment.

    Strict prompt structure (numbered I/O) is used to minimise tag
    contamination across comments in the same batch.
    """
    llm = build_llm()

    # Build a numbered input block so the model maps output back to input cleanly
    numbered_input = "\n".join(
        f"{i+1}. {c}" for i, c in enumerate(comments)
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_SYSTEM_PROMPT),
        ("human", "{numbered_comments}")
    ])

    chain = prompt | llm
    response = chain.invoke({"numbered_comments": numbered_input})

    # Parse raw JSON response — strip fences defensively
    raw = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    parsed = json.loads(raw)

    return [
        StructuredFeedbackSchema(
            intents=item["intents"],
            tags=item["tags"]
        )
        for item in parsed["results"]
    ]


def extract_metadata_in_batches(
    all_comments: List[str],
) -> List[StructuredFeedbackSchema]:
    """
    Chunks the full comment list into BATCH_SIZE groups and calls
    extract_metadata_batch() for each chunk.
    """
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE):
        chunk = all_comments[start : start + BATCH_SIZE]
        print(f"  Processing batch [{start+1}–{start+len(chunk)}] ...")
        batch_results = extract_metadata_batch(chunk)
        all_results.extend(batch_results)
    return all_results


# ── SPLADE Model Init ─────────────────────────────────────────────────────────

print("Loading SPLADE model layers ...")
tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3-distilbert")
model     = AutoModelForMaskedLM.from_pretrained("naver/splade-v3-distilbert")
model.eval()


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
    """Standard SPLADE forward pass → sparse {indices, values} dict."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True,
        truncation=True, max_length=512
    )
    with torch.no_grad():
        output = model(**inputs)

    relu_log   = torch.log(1 + torch.relu(output.logits))
    sparse_vec, _ = torch.max(relu_log, dim=1)
    sparse_vec = sparse_vec.squeeze()

    non_zero_indices = sparse_vec.nonzero().squeeze()
    if non_zero_indices.dim() == 0:
        non_zero_indices = non_zero_indices.unsqueeze(0)

    return {
        "indices": non_zero_indices.tolist(),
        "values":  sparse_vec[non_zero_indices].tolist()
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


# ── Query: Parallel Intent + Tag Filter ───────────────────────────────────────

def search_parallel_filters(
    keyword:        str,
    target_intent:  Optional[str] = None,
    target_tag:     Optional[str] = None,
    top_k:          int = 3
) -> list:
    """
    Runs two searches in parallel against the same SPLADE query vector:
      - Stream A: filtered by intent (if provided)
      - Stream B: filtered by tag   (if provided)

    Results from both streams are merged by doc ID, deduped, and
    re-ranked by best score. Documents matching BOTH filters naturally
    score higher because they appear in both result sets.

    Args:
        keyword:       Single word or short phrase from the user.
        target_intent: Canonical intent string to filter on (optional).
        target_tag:    Tag phrase to filter on (optional).
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

    seen    = {}   # doc_id → best hit dict
    streams = []

    # Build stream A — intent filter
    if target_intent:
        intent_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="intents",
                    match=models.MatchValue(value=target_intent)
                )
            ]
        )
        streams.append(("intent", intent_filter))

    # Build stream B — tag filter
    if target_tag:
        tag_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="tags",
                    match=models.MatchValue(value=target_tag)
                )
            ]
        )
        streams.append(("tag", tag_filter))

    # Fallback: no filters provided → single unfiltered search
    if not streams:
        streams.append(("unfiltered", None))

    # Execute both streams and merge
    for stream_label, search_filter in streams:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=sparse_query_vector,
            query_filter=search_filter,
            limit=top_k
        )
        for hit in results:
            doc_id = hit.id
            # Keep the highest score if a doc appears in both streams
            if doc_id not in seen or hit.score > seen[doc_id]["score"]:
                seen[doc_id] = {
                    "score":    hit.score,
                    "text":     hit.payload["original_text"],
                    "intents":  hit.payload["intents"],
                    "tags":     hit.payload["tags"],
                    "streams":  set()
                }
            seen[doc_id]["streams"].add(stream_label)

    # Sort merged results by score descending
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

    # ── Step 1: Batch LLM Extraction ─────────────────────────────────────────
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
    print("\n=== Step 2: Parallel Intent + Tag Filtered Search ===")

    keyword       = "prior auth"
    intent_filter = "Coverage Inquiry"
    tag_filter    = "prior auth denial"   # example tag phrase from extraction

    results = search_parallel_filters(
        keyword=keyword,
        target_intent=intent_filter,
        target_tag=tag_filter,
        top_k=3
    )

    print(f"\nKeyword : '{keyword}'")
    print(f"Filters : intent='{intent_filter}' | tag='{tag_filter}'\n")

    for rank, match in enumerate(results, start=1):
        print(f"[Rank {rank}]  Score: {match['score']:.4f}  |  Matched streams: {match['streams']}")
        print(f"  Text    : {match['text'][:100]}...")
        print(f"  Intents : {match['intents']}")
        print(f"  Tags    : {match['tags']}\n")
