# =============================================================================
# pipeline_v1_batch_filter.py
#
# ARCHITECTURE: Semantic Fingerprint Ingest + HyDE Query
#               + Tiered Threshold Retrieval + Parallel LLM Classification
#
# Ingest phase:
#   - Claude generates a semantic fingerprint per comment:
#     query_variants, implicit_concepts, related_scenarios
#   - Multi-topic comments get independent coverage per topic
#   - Each phrase embedded individually (exploded index)
#   - No abbreviations — all terms expanded at ingest
#
# Query phase — three-tier result pipeline:
#
#   Tier 1 | HIGH CONFIDENCE  (score >= UPPER_THRESHOLD)
#             Auto-included. Embedding similarity strong enough
#             that LLM classification is unnecessary.
#
#   Tier 2 | LLM VERIFIED     (LOWER_THRESHOLD <= score < UPPER_THRESHOLD)
#             Ambiguous zone. Passed to Claude in parallel batches of 20
#             for binary include/exclude decision.
#             Only this tier incurs LLM cost at query time.
#
#   Tier 3 | EXCLUDED          (score < LOWER_THRESHOLD)
#             Auto-excluded. Similarity too low to be relevant.
#             Never shown, never sent to LLM.
#
# Threshold tuning:
#   UPPER_THRESHOLD : start at 0.75, raise if too many auto-includes are noisy
#   LOWER_THRESHOLD : start at 0.45, lower if relevant comments are being missed
#   Tune by inspecting score distributions on real queries after first run.
#
# Auth:
#   OpenAI-compatible Databricks client for LLM and embeddings.
#   Base URL, LLM model, embedding model collected at startup.
# =============================================================================

# %pip install openai "qdrant-client>=1.9.0" pydantic
# dbutils.library.restartPython()


# ── Imports ───────────────────────────────────────────────────────────────────

import os
import json
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
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

COLLECTION_NAME  = "health_feedback_fingerprint_index"
BATCH_SIZE_INGEST = 5     # comments per LLM call at ingest
HYDE_COUNT        = 3     # hypothetical comments generated per query
HYDE_LIMIT        = 100   # Qdrant results per HyDE vector (wide net for recall)
CLASSIFY_BATCH    = 20    # comments per LLM classification call

# ── Tiered thresholds ─────────────────────────────────────────────────────────
# Tune these after inspecting score distributions on real queries.
# Run print_score_distribution() after a few searches to calibrate.
UPPER_THRESHOLD = 0.75    # at or above → AUTO-INCLUDE  (High Confidence)
LOWER_THRESHOLD = 0.45    # below       → AUTO-EXCLUDE  (not shown, not sent to LLM)
                           # between     → LLM decides   (binary include/exclude)


# ── Databricks Client ─────────────────────────────────────────────────────────

DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

db_client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=DATABRICKS_BASE_URL
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class TopicBlock(BaseModel):
    topic_label      : str
    query_variants   : List[str]
    implicit_concepts: List[str]
    related_scenarios: List[str]

class FingerprintSchema(BaseModel):
    topics       : List[TopicBlock]
    severity     : str
    journey_stages: List[str]

class TagSchema(BaseModel):
    """Fallback schema used if fingerprint generation fails."""
    tags: List[str] = Field(
        description=(
            "2-6 descriptive 2-3 word phrases. "
            "Always expand abbreviations to full terms. "
            "Never use acronym-only tags."
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
    cleaned = raw.strip()
    for fence in ["```json", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            break
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


# ── Semantic Fingerprint Extraction (Ingest) ──────────────────────────────────

FINGERPRINT_SYSTEM_PROMPT = """\
You are an expert healthcare member experience analyst.
Given a health insurance member feedback comment, generate a semantic
fingerprint that maximises future retrieval accuracy.

CRITICAL — abbreviation rules:
  Always expand ALL abbreviations to full clinical terms.
  "prior authorization" not "prior auth", "explanation of benefits" not "EOB",
  "out of pocket maximum" not "OOP max", "procedure code" not "CPT code".
  Never use acronym-only phrases anywhere in the output.

CRITICAL — multi-topic handling:
  If the comment contains more than one distinct problem, create a separate
  topic block for EACH problem. Give equal coverage to all topics.
  Do not let the dominant complaint overshadow secondary ones.

For each topic block generate:
  topic_label       : 3-5 word label for this specific problem
  query_variants    : 8-10 different short phrases (2-4 words each) that a
                      person might type when searching for this problem.
                      Cover clinical terms, colloquial language, and synonyms.
  implicit_concepts : 4-5 concepts implied by this complaint but not
                      explicitly stated (e.g. downstream actions, related
                      processes, member sentiment signals)
  related_scenarios : 4-5 adjacent problems that share the same root cause
                      or member journey stage

Also generate at document level:
  severity          : "low", "medium", "high", or "critical"
  journey_stages    : array of applicable stages from:
                      ["prior authorization", "claims", "billing",
                       "digital", "customer service", "pharmacy",
                       "enrollment", "referral", "appeals"]

Return ONLY valid JSON matching this structure:
{
  "topics": [
    {
      "topic_label": "...",
      "query_variants": ["...", ...],
      "implicit_concepts": ["...", ...],
      "related_scenarios": ["...", ...]
    }
  ],
  "severity": "...",
  "journey_stages": ["...", ...]
}
No preamble, no markdown fences.
"""

BATCH_FINGERPRINT_SYSTEM_PROMPT = """\
You are an expert healthcare member experience analyst.
You will receive a numbered list of member feedback comments.
For EACH comment independently generate a semantic fingerprint.

Follow ALL rules below for every comment:

ABBREVIATION RULES — always expand:
  "prior authorization" not "prior auth", "explanation of benefits" not "EOB",
  "out of pocket maximum" not "OOP max", "procedure code" not "CPT code".
  Never use acronym-only phrases.

MULTI-TOPIC RULE — if a comment contains more than one distinct problem,
  create a separate topic block per problem with equal coverage.

Per topic block:
  topic_label       : 3-5 word label
  query_variants    : 8-10 search phrases (2-4 words, varied vocabulary)
  implicit_concepts : 4-5 implied concepts not explicitly stated
  related_scenarios : 4-5 adjacent problems at the same journey stage

Per document:
  severity          : "low" / "medium" / "high" / "critical"
  journey_stages    : from ["prior authorization","claims","billing",
                      "digital","customer service","pharmacy",
                      "enrollment","referral","appeals"]

Return a JSON object with a single key "results" — ordered array,
one fingerprint object per input comment.
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def extract_fingerprints_batch(
    comments: List[str]
) -> List[FingerprintSchema]:
    """
    Extracts semantic fingerprints for up to BATCH_SIZE_INGEST comments
    in a single Claude call. Returns one FingerprintSchema per comment.
    Falls back to empty topic block on parse failure so pipeline continues.
    """
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    raw      = llm_call(BATCH_FINGERPRINT_SYSTEM_PROMPT, numbered)

    try:
        parsed = parse_json_response(raw)
        results = []
        for item in parsed["results"]:
            topics = [
                TopicBlock(
                    topic_label      = t.get("topic_label", "general"),
                    query_variants   = t.get("query_variants", []),
                    implicit_concepts= t.get("implicit_concepts", []),
                    related_scenarios= t.get("related_scenarios", [])
                )
                for t in item.get("topics", [])
            ]
            results.append(FingerprintSchema(
                topics        = topics,
                severity      = item.get("severity", "medium"),
                journey_stages= item.get("journey_stages", [])
            ))
        return results

    except Exception as e:
        print(f"  [WARN] Fingerprint parse failed: {e} — using empty fallback")
        return [
            FingerprintSchema(topics=[], severity="medium", journey_stages=[])
            for _ in comments
        ]


def extract_fingerprints_in_batches(
    all_comments: List[str]
) -> List[FingerprintSchema]:
    all_results = []
    for start in range(0, len(all_comments), BATCH_SIZE_INGEST):
        chunk = all_comments[start : start + BATCH_SIZE_INGEST]
        print(f"  Fingerprinting batch [{start+1}–{start+len(chunk)}] ...")
        all_results.extend(extract_fingerprints_batch(chunk))
    return all_results


# ── HyDE Query Expansion ──────────────────────────────────────────────────────

HYDE_SYSTEM_PROMPT = f"""\
You are a healthcare member experience specialist.
Given a search keyword related to a health insurance problem, write
{HYDE_COUNT} realistic member feedback comments a real member would submit.

Each comment should:
  - Be 2-4 sentences in natural member language
  - Cover a slightly different aspect or scenario of the problem
  - Sound like a genuine NPS survey verbatim response
  - Use varied vocabulary — do not repeat the same phrases across comments

Return JSON: {{"comments": ["comment1", "comment2", "comment3"]}}
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def generate_hyde_comments(keyword: str) -> List[str]:
    print(f"\n  [HyDE] Generating hypothetical comments for '{keyword}' ...")
    raw     = llm_call(HYDE_SYSTEM_PROMPT, f"Keyword: {keyword}")
    parsed  = parse_json_response(raw)
    comments = parsed.get("comments", [keyword])
    for i, c in enumerate(comments, 1):
        print(f"  [HyDE {i}] {c[:90]}...")
    return comments


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    response = db_client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    vec = response.data[0].embedding
    return vec if isinstance(vec, list) else list(vec)


# ── Qdrant Init ───────────────────────────────────────────────────────────────

print("Resolving embedding dimension ...")
_test_vector = embed_text("test")
EMBED_DIM    = len(_test_vector)
print(f"  Embedding dimension : {EMBED_DIM}")

print("Initialising Qdrant ...")
qdrant = QdrantClient(":memory:")

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

# Full-text index on phrase field for Stream B exact matching
qdrant.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="phrase",
    field_schema=models.TextIndexParams(
        type=models.TextIndexType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        max_token_len=30,
        lowercase=True
    )
)
print(f"  Collection '{COLLECTION_NAME}' ready (dim={EMBED_DIM})")


# ── Indexing ──────────────────────────────────────────────────────────────────

_point_id_counter = 0

def index_document(
    doc_id     : int,
    text       : str,
    fingerprint: FingerprintSchema
):
    """
    Exploded index — one Qdrant point per phrase across all topic blocks.
    Each point stores:
      vector  : embedding of the individual phrase
      payload : doc_id, original text, phrase, phrase_type, topic_label,
                all_topics summary, severity, journey_stages

    phrase_type distinguishes query_variants / implicit_concepts /
    related_scenarios so you can filter by type if needed later.
    """
    global _point_id_counter
    points = []

    all_topics_summary = [t.topic_label for t in fingerprint.topics]

    for topic in fingerprint.topics:
        phrase_groups = [
            ("query_variant",    topic.query_variants),
            ("implicit_concept", topic.implicit_concepts),
            ("related_scenario", topic.related_scenarios),
        ]
        for phrase_type, phrases in phrase_groups:
            for phrase in phrases:
                if not phrase.strip():
                    continue
                _point_id_counter += 1
                vector = embed_text(phrase)
                points.append(
                    models.PointStruct(
                        id=_point_id_counter,
                        vector=vector,
                        payload={
                            "doc_id"          : doc_id,
                            "original_text"   : text,
                            "phrase"          : phrase,
                            "phrase_type"     : phrase_type,
                            "topic_label"     : topic.topic_label,
                            "all_topics"      : all_topics_summary,
                            "severity"        : fingerprint.severity,
                            "journey_stages"  : fingerprint.journey_stages
                        }
                    )
                )

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    print(
        f"  Doc {doc_id} indexed — "
        f"{len(fingerprint.topics)} topic(s), "
        f"{len(points)} phrase points"
    )


# ── Broad Retrieval ───────────────────────────────────────────────────────────

def retrieve_candidates(keyword: str) -> List[Dict]:
    """
    Stage 1 — HyDE expansion + broad Qdrant retrieval.

    Generates HYDE_COUNT hypothetical comments, embeds each,
    and runs a wide search (limit=HYDE_LIMIT, no score floor).
    Results merged by doc_id keeping the best embedding score.

    Returns flat list of candidate dicts sorted by score DESC.
    No threshold applied here — thresholds are applied in tier_candidates().
    """
    hyde_comments = generate_hyde_comments(keyword)
    seen: Dict[int, Dict] = {}

    for i, comment in enumerate(hyde_comments):
        vector   = embed_text(comment)
        response = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=HYDE_LIMIT,
            with_payload=True
        )
        print(f"  [Retrieval] HyDE {i+1} → {len(response.points)} raw hits")

        for hit in response.points:
            doc_id = hit.payload["doc_id"]
            if doc_id not in seen or hit.score > seen[doc_id]["embedding_score"]:
                seen[doc_id] = {
                    "doc_id"         : doc_id,
                    "embedding_score": hit.score,
                    "original_text"  : hit.payload["original_text"],
                    "matched_phrase" : hit.payload["phrase"],
                    "phrase_type"    : hit.payload["phrase_type"],
                    "topic_label"    : hit.payload["topic_label"],
                    "all_topics"     : hit.payload["all_topics"],
                    "severity"       : hit.payload["severity"],
                    "journey_stages" : hit.payload["journey_stages"]
                }

    candidates = sorted(
        seen.values(),
        key=lambda x: x["embedding_score"],
        reverse=True
    )
    print(f"  [Retrieval] {len(candidates)} unique docs after merge\n")
    return candidates


# ── Tiered Threshold Split ────────────────────────────────────────────────────

def tier_candidates(
    candidates: List[Dict]
) -> Tuple[List[Dict], List[Dict], int]:
    """
    Splits candidates into three tiers by embedding score:

      Tier 1 (high_confidence) : score >= UPPER_THRESHOLD
        Auto-included. No LLM call needed.

      Tier 2 (ambiguous)       : LOWER_THRESHOLD <= score < UPPER_THRESHOLD
        Sent to LLM for binary include/exclude.

      Tier 3 (excluded)        : score < LOWER_THRESHOLD
        Dropped silently. Never shown, never sent to LLM.

    Returns (high_confidence, ambiguous, excluded_count)
    """
    high_confidence = []
    ambiguous       = []
    excluded_count  = 0

    for c in candidates:
        score = c["embedding_score"]
        if score >= UPPER_THRESHOLD:
            high_confidence.append(c)
        elif score >= LOWER_THRESHOLD:
            ambiguous.append(c)
        else:
            excluded_count += 1

    print(f"  [Tiers] High confidence : {len(high_confidence)} "
          f"(score >= {UPPER_THRESHOLD})")
    print(f"  [Tiers] Ambiguous       : {len(ambiguous)} "
          f"({LOWER_THRESHOLD} <= score < {UPPER_THRESHOLD})")
    print(f"  [Tiers] Excluded        : {excluded_count} "
          f"(score < {LOWER_THRESHOLD})\n")

    return high_confidence, ambiguous, excluded_count


# ── LLM Binary Classification ─────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = """\
You are a healthcare member experience analyst.
You will receive a search query and a batch of member feedback comments.

For EACH comment decide independently:
  RELEVANT   — the comment is genuinely about the query topic
               (directly or as a closely related concept)
  NOT_RELEVANT — the comment is not meaningfully about the query topic

Be consistent. Judge each comment independently against the query.

Return JSON:
{
  "decisions": [
    {"doc_id": <int>, "decision": "RELEVANT" or "NOT_RELEVANT"},
    ...
  ]
}
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def classify_batch(batch: List[Dict], keyword: str) -> List[Dict]:
    """
    Classifies one batch of ambiguous candidates as RELEVANT / NOT_RELEVANT.
    Returns the subset decided RELEVANT with llm_verified=True in payload.
    """
    formatted = "\n\n".join(
        f"[doc_id={c['doc_id']}]\n{c['original_text']}"
        for c in batch
    )
    user_msg = f'Query: "{keyword}"\n\nComments:\n{formatted}'
    raw      = llm_call(CLASSIFY_SYSTEM_PROMPT, user_msg)

    try:
        parsed    = parse_json_response(raw)
        decisions = {
            d["doc_id"]: d["decision"]
            for d in parsed.get("decisions", [])
        }
    except Exception as e:
        print(f"  [WARN] Classification parse failed: {e} — excluding batch")
        decisions = {}

    verified = []
    for c in batch:
        decision = decisions.get(c["doc_id"], "NOT_RELEVANT")
        if decision == "RELEVANT":
            c["llm_verified"] = True
            verified.append(c)

    return verified


def classify_ambiguous_parallel(
    ambiguous: List[Dict],
    keyword  : str
) -> List[Dict]:
    """
    Splits ambiguous candidates into batches of CLASSIFY_BATCH and
    runs all classification calls in parallel via ThreadPoolExecutor.
    Returns flat list of LLM-verified relevant candidates.
    """
    if not ambiguous:
        return []

    batches = [
        ambiguous[i : i + CLASSIFY_BATCH]
        for i in range(0, len(ambiguous), CLASSIFY_BATCH)
    ]

    print(f"  [Classify] {len(ambiguous)} ambiguous → "
          f"{len(batches)} parallel batches ...")

    all_verified = []

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(classify_batch, batch, keyword): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            result = future.result()
            all_verified.extend(result)

    # Sort by embedding score descending within verified set
    all_verified.sort(key=lambda x: x["embedding_score"], reverse=True)

    print(f"  [Classify] {len(all_verified)} / {len(ambiguous)} "
          f"ambiguous passed LLM filter\n")

    return all_verified


# ── Score Distribution Utility ────────────────────────────────────────────────

def print_score_distribution(candidates: List[Dict], keyword: str):
    """
    Prints a histogram of embedding scores across all retrieved candidates.
    Run this on a few real queries to calibrate UPPER_THRESHOLD and
    LOWER_THRESHOLD for your specific data and embedding model.

    Look for natural gaps in the distribution — those are where to set
    the thresholds. A gap between 0.70-0.75 suggests UPPER=0.72 etc.
    """
    if not candidates:
        print("  No candidates to analyse.")
        return

    buckets = {}
    for c in candidates:
        bucket = round(c["embedding_score"], 1)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    print(f"\n  Score distribution for query '{keyword}':")
    for score in sorted(buckets.keys(), reverse=True):
        bar   = "█" * buckets[score]
        tier  = (
            "← AUTO-INCLUDE"  if score >= UPPER_THRESHOLD else
            "← LLM ZONE"      if score >= LOWER_THRESHOLD else
            "← AUTO-EXCLUDE"
        )
        print(f"  {score:.1f} | {bar:<30} {buckets[score]:>3}  {tier}")
    print()


# ── Full Search Pipeline ──────────────────────────────────────────────────────

def search(keyword: str) -> Dict:
    """
    Full three-tier search pipeline:

      1. HyDE expansion + broad Qdrant retrieval
      2. Tier split by embedding score
         - High confidence → auto-include
         - Ambiguous       → parallel LLM binary classification
         - Below floor     → auto-exclude
      3. Merge high_confidence + llm_verified into final result set
      4. Sort by (tier, embedding_score DESC)

    Returns dict with:
      high_confidence : auto-included results
      llm_verified    : ambiguous results that passed LLM filter
      excluded_count  : number of docs below lower threshold
      total_relevant  : len(high_confidence) + len(llm_verified)
    """
    print(f"\n{'='*62}")
    print(f"  SEARCH: '{keyword}'")
    print(f"{'='*62}\n")

    # Stage 1 — broad retrieval
    candidates = retrieve_candidates(keyword)

    # Optional: uncomment to calibrate thresholds on real data
    # print_score_distribution(candidates, keyword)

    # Stage 2 — tier split
    high_confidence, ambiguous, excluded_count = tier_candidates(candidates)

    # Stage 3 — LLM classification for ambiguous tier only
    llm_verified = classify_ambiguous_parallel(ambiguous, keyword)

    return {
        "keyword"        : keyword,
        "high_confidence": high_confidence,
        "llm_verified"   : llm_verified,
        "excluded_count" : excluded_count,
        "total_relevant" : len(high_confidence) + len(llm_verified)
    }


# ── Result Rendering ──────────────────────────────────────────────────────────

def render_results(results: Dict):
    """
    Renders final results with tier labels.
    HIGH CONFIDENCE shown first, LLM VERIFIED shown second.
    Each result shows embedding score, matched phrase, topic, journey stage.
    """
    keyword    = results["keyword"]
    hc         = results["high_confidence"]
    lv         = results["llm_verified"]
    excl_count = results["excluded_count"]
    total      = results["total_relevant"]
    sep        = "─" * 62

    print(f"\n{'='*62}")
    print(f"  Results for: '{keyword}'")
    print(f"  Total relevant: {total}  "
          f"(HC: {len(hc)}  LLM: {len(lv)}  Excluded: {excl_count})")
    print(f"{'='*62}")

    # ── HIGH CONFIDENCE section ───────────────────────────────────────────────
    print(f"\n  ▌HIGH CONFIDENCE  (embedding >= {UPPER_THRESHOLD})")
    print(f"  {sep}")
    if not hc:
        print("  None.")
    for rank, r in enumerate(hc, start=1):
        print(f"\n  [{rank}]  Score: {r['embedding_score']:.4f}  "
              f"Doc: {r['doc_id']}  Severity: {r['severity']}")
        print(f"  Topic          : {r['topic_label']}")
        print(f"  Journey stages : {r['journey_stages']}")
        print(f"  Matched phrase : '{r['matched_phrase']}' "
              f"[{r['phrase_type']}]")
        print(f"  Text           : {r['original_text'][:110]}...")

    # ── LLM VERIFIED section ──────────────────────────────────────────────────
    print(f"\n  ▌LLM VERIFIED  "
          f"({LOWER_THRESHOLD} <= embedding < {UPPER_THRESHOLD}, "
          f"passed Claude filter)")
    print(f"  {sep}")
    if not lv:
        print("  None passed LLM filter.")
    for rank, r in enumerate(lv, start=1):
        print(f"\n  [{rank}]  Score: {r['embedding_score']:.4f}  "
              f"Doc: {r['doc_id']}  Severity: {r['severity']}")
        print(f"  Topic          : {r['topic_label']}")
        print(f"  Journey stages : {r['journey_stages']}")
        print(f"  Matched phrase : '{r['matched_phrase']}' "
              f"[{r['phrase_type']}]")
        print(f"  Text           : {r['original_text'][:110]}...")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  {sep}")
    print(f"  Auto-excluded (below {LOWER_THRESHOLD}): {excl_count} docs")
    ambiguous_total = len(results["llm_verified"]) + len(lv)
    llm_batches     = (ambiguous_total // CLASSIFY_BATCH) + (1 if ambiguous_total % CLASSIFY_BATCH else 0)
    print(f"  LLM calls made : {llm_batches} batch(es) of {CLASSIFY_BATCH}")
    print(f"{'='*62}\n")


# ── Main Pipeline ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    raw_feedbacks = [
        "I was on hold for two full hours, and when the agent finally picked "
        "up, they told me my prior authorization for the knee MRI was rejected.",

        "The portal kept crashing when I tried to upload my claim documents, "
        "and on top of that my claim has been sitting in processing for 6 weeks "
        "with no updates.",                    # ← multi-topic: digital + claims

        "The medical treatment was fine, but the billing department sent me a "
        "huge invoice because they typed the procedure code wrong.",

        "My specialist referral was denied without any explanation and I cannot "
        "reach anyone in the approvals department.",

        "I received two explanations of benefits for the same procedure with "
        "different amounts and now I do not know which one to pay.",
    ]

    # ── Step 1: Fingerprint Extraction + Indexing ─────────────────────────────
    print("=== Step 1: Semantic Fingerprint Extraction ===\n")
    fingerprints = extract_fingerprints_in_batches(raw_feedbacks)

    print("\n=== Indexing Documents ===\n")
    for idx, (comment, fp) in enumerate(
        zip(raw_feedbacks, fingerprints), start=1
    ):
        print(f"  Doc {idx}: {comment[:70]}...")
        for t in fp.topics:
            print(f"    Topic: {t.topic_label}")
        index_document(doc_id=idx, text=comment, fingerprint=fp)

    print(f"\n  Total points in index: {_point_id_counter}\n")

    # ── Step 2: Tiered Search ─────────────────────────────────────────────────
    print("=== Step 2: Tiered Threshold Search ===")

    for keyword in ["prior auth", "claim delay", "portal error"]:
        results = search(keyword)
        render_results(results)

        # Uncomment to inspect score distributions and tune thresholds:
        # all_candidates = retrieve_candidates(keyword)
        # print_score_distribution(all_candidates, keyword)
