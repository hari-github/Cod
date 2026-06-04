# =============================================================================
# pipeline_v2_hyde_combined.py
#
# ARCHITECTURE: HyDE Query Expansion + Batch LLM Extraction
#               + Parallel Metadata-Filtered Sparse Search
#
# Design rationale:
#   - HyDE expands a single sparse keyword into a richer hypothetical document,
#     recovering the recall advantage of dense query expansion within SPLADE.
#   - Batch LLM extraction (5 comments/call) cuts ingest API cost.
#   - Parallel intent + tag pre-filters add precision without sacrificing
#     the recall benefit that HyDE provides.
#   - The two streams (intent-filtered, tag-filtered) are merged by score;
#     documents matching both filters naturally rank higher.
# =============================================================================

# %pip install torch transformers qdrant-client langchain-core langchain-openai pydantic
# dbutils.library.restartPython()


# ── Imports ───────────────────────────────────────────────────────────────────

import json
import torch
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client import models


# ── Configuration ─────────────────────────────────────────────────────────────

DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"
DATABRICKS_TOKEN = dbutils.secrets.get(scope="<your-scope>", key="databricks-token")
MODEL_ENDPOINT   = "databricks-claude-sonnet"
COLLECTION_NAME  = "structured_health_feedback_v2"
BATCH_SIZE       = 5


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class StructuredFeedbackSchema(BaseModel):
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


# ── LLM Init ─────────────────────────────────────────────────────────────────

def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_ENDPOINT,
        temperature=0.0,
        openai_api_key=DATABRICKS_TOKEN,
        openai_api_base=f"{DATABRICKS_HOST}/serving-endpoints",
    )


# ── Batch Metadata Extraction (Ingest Phase) ──────────────────────────────────

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
    Sends up to BATCH_SIZE comments in a single Claude call and returns
    one StructuredFeedbackSchema per comment in input order.
    """
    llm = build_llm()
    numbered_input = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))

    prompt = ChatPromptTemplate.from_messages([
        ("system", BATCH_SYSTEM_PROMPT),
        ("human", "{numbered_comments}")
    ])
    chain  = prompt | llm
    response = chain.invoke({"numbered_comments": numbered_input})

    raw    = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    parsed = json.loads(raw)

    return [
        StructuredFeedbackSchema(intents=item["intents"], tags=item["tags"])
        for item in parsed["results"]
    ]


def extract_metadata_in_batches(all_comments: List[str]) -> List[StructuredFeedbackSchema]:
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE):
        chunk = all_comments[start : start + BATCH_SIZE]
        print(f"  Extracting batch [{start+1}–{start+len(chunk)}] ...")
        all_results.extend(extract_metadata_batch(chunk))
    return all_results


# ── HyDE Query Expansion (Query Phase) ───────────────────────────────────────
#
# Instead of encoding the raw keyword directly, we ask Claude to write a
# short hypothetical feedback comment that a real member would leave if they
# had the problem described by the keyword.
#
# Why this helps SPLADE specifically:
#   SPLADE fires on tokens that are literally present in the text. A keyword
#   like "prior auth" contains only 2 tokens. A HyDE-expanded document will
#   contain the full vocabulary a real complaint uses — "denied", "approval",
#   "specialist", "authorization" — all of which SPLADE will weight and match
#   against the index. This closes the vocabulary gap without needing a
#   separate dense encoder.

HYDE_SYSTEM_PROMPT = """\
You are a healthcare member experience specialist.
Given a short search keyword or phrase describing a health insurance problem,
write ONE realistic, concise customer feedback comment (2–4 sentences) that
a real health insurance member would submit.
Use natural language and specific details a member would mention.
Return ONLY the comment text — no preamble, labels, or quotes.
"""

def expand_query_with_hyde(keyword: str) -> str:
    """
    Calls Claude to generate a hypothetical feedback comment from the keyword.
    The generated text is then encoded with SPLADE for richer token coverage.
    """
    llm = build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", HYDE_SYSTEM_PROMPT),
        ("human", "Keyword: {keyword}")
    ])
    chain    = prompt | llm
    response = chain.invoke({"keyword": keyword})
    return response.content.strip()


# ── SPLADE Model Init ─────────────────────────────────────────────────────────

print("Loading SPLADE model layers ...")
tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3-distilbert")
splade    = AutoModelForMaskedLM.from_pretrained("naver/splade-v3-distilbert")
splade.eval()


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
        output = splade(**inputs)

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
    sparse_data = compute_splade_vector(text)
    payload = {"original_text": text, "intents": intents, "tags": tags}
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


# ── Query: HyDE Expansion + Parallel Intent + Tag Filter ─────────────────────

def search_hyde_parallel_filters(
    keyword:        str,
    target_intent:  Optional[str] = None,
    target_tag:     Optional[str] = None,
    top_k:          int = 3,
    verbose:        bool = True
) -> list:
    """
    Full combined query pipeline:

      1. HyDE — keyword → Claude → hypothetical feedback comment
      2. SPLADE encode the expanded document (richer token coverage)
      3. Stream A — SPLADE search with intent pre-filter
      4. Stream B — SPLADE search with tag pre-filter
      5. Merge both streams by doc ID, keep best score per doc
      6. Re-rank merged set by score descending

    Args:
        keyword:       Single word or short phrase from user.
        target_intent: Canonical intent to pre-filter on (optional).
        target_tag:    Tag phrase to pre-filter on (optional).
        top_k:         Number of final results to return.
        verbose:       If True, prints the HyDE-generated document.
    """

    # Step 1 — HyDE Expansion
    print(f"\n[HyDE] Expanding keyword: '{keyword}' ...")
    hypothetical_doc = expand_query_with_hyde(keyword)
    if verbose:
        print(f"[HyDE] Generated document:\n  \"{hypothetical_doc}\"\n")

    # Step 2 — SPLADE encode the hypothetical document
    query_data = compute_splade_vector(hypothetical_doc)

    sparse_query_vector = models.NamedSparseVector(
        name="splade_vector",
        vector=models.SparseVector(
            indices=query_data["indices"],
            values=query_data["values"]
        )
    )

    # Step 3 + 4 — Build parallel filter streams
    seen    = {}
    streams = []

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

    # Fallback: no filters → single unfiltered search on HyDE vector
    if not streams:
        streams.append(("unfiltered", None))

    # Step 5 — Execute and merge
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

    # Step 6 — Re-rank merged results
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

    # ── Step 2: HyDE + Parallel Filtered Search ───────────────────────────────
    print("\n=== Step 2: HyDE Expansion + Parallel Intent/Tag Filtered Search ===")

    keyword       = "prior auth"
    intent_filter = "Coverage Inquiry"
    tag_filter    = "prior auth denial"

    results = search_hyde_parallel_filters(
        keyword=keyword,
        target_intent=intent_filter,
        target_tag=tag_filter,
        top_k=3,
        verbose=True
    )

    print(f"Keyword : '{keyword}'")
    print(f"Filters : intent='{intent_filter}' | tag='{tag_filter}'\n")

    for rank, match in enumerate(results, start=1):
        print(f"[Rank {rank}]  Score: {match['score']:.4f}  |  Matched streams: {match['streams']}")
        print(f"  Text    : {match['text'][:100]}...")
        print(f"  Intents : {match['intents']}")
        print(f"  Tags    : {match['tags']}\n")
