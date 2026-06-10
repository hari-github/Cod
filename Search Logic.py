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
#   Tier 1 | HIGH CONFIDENCE  (final_score >= upper_threshold)
#             Auto-included. Weighted embedding similarity strong enough
#             that LLM classification is unnecessary.
#
#   Tier 2 | LLM VERIFIED     (lower_threshold <= final_score < upper_threshold)
#             Ambiguous zone. Passed to Claude in parallel batches of 20
#             for binary include/exclude decision.
#             Only this tier incurs LLM cost at query time.
#
#   Tier 3 | EXCLUDED          (final_score < lower_threshold)
#             Auto-excluded. Similarity too low to be relevant.
#             Never shown, never sent to LLM.
#
# Retrieval improvements (v2):
#   Option 1 — Reciprocal Rank Fusion (RRF) replaces best-score-wins.
#              Docs hit by multiple HyDE vectors accumulate score and rank
#              higher than one-hit wonders at the same raw cosine similarity.
#   Option 2 — Stream B keyword search. The full-text phrase index (already
#              built at ingest) is now actually queried. Keyword hits boost
#              existing candidates and add new ones at the LLM zone floor.
#   Option 3 — LLM classifier sees retrieval context. Each comment is shown
#              with the matched phrase and topic that caused it to be surfaced,
#              giving the model the same signal the retrieval system used.
#   Option 4 — phrase_type weighting. query_variant hits accumulate more RRF
#              score than implicit_concept or related_scenario hits; the best
#              match's weight also scales the final_score used for tiering.
#   Option 5 — Adaptive thresholds per query. Tier boundaries auto-calibrate
#              from the score distribution, within ±0.10 of static defaults.
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

COLLECTION_NAME   = "health_feedback_fingerprint_index"
BATCH_SIZE_INGEST = 5     # comments per LLM call at ingest
HYDE_COUNT        = 3     # hypothetical comments generated per query
HYDE_LIMIT        = 100   # Qdrant results per HyDE vector (wide net for recall)
CLASSIFY_BATCH    = 20    # comments per LLM classification call

# ── Tiered thresholds ─────────────────────────────────────────────────────────
# Static defaults — adaptive thresholds stay within ±0.10 of these.
# Run print_score_distribution() after a few searches to calibrate.
UPPER_THRESHOLD = 0.75    # at or above → AUTO-INCLUDE  (High Confidence)
LOWER_THRESHOLD = 0.45    # below       → AUTO-EXCLUDE  (not shown, not sent to LLM)
                           # between     → LLM decides   (binary include/exclude)

# ── Option 4: phrase_type scoring weights ─────────────────────────────────────
# Applied during RRF accumulation and to embedding_score to compute final_score.
# query_variant is the strongest retrieval signal; related_scenario is weakest.
TYPE_WEIGHT: Dict[str, float] = {
    "query_variant":    1.00,
    "implicit_concept": 0.90,
    "related_scenario": 0.80,
}

# ── Option 2: keyword boost ───────────────────────────────────────────────────
# Added to rrf_score for any doc whose indexed phrases contain the raw keyword.
KEYWORD_BOOST = 0.05


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

ABBREVIATION RULES:
  In topic_label, implicit_concepts, and related_scenarios — always expand:
  "prior authorization" not "prior auth", "explanation of benefits" not "EOB",
  "out of pocket maximum" not "OOP max", "procedure code" not "CPT code".
  EXCEPTION — query_variants should include BOTH expanded forms AND the
  informal short forms members actually type ("prior auth", "auth denied"),
  since search coverage requires both. Acronym-only phrases ("EOB", "PA")
  are still never allowed anywhere.

MULTI-TOPIC RULE — if a comment contains more than one distinct problem,
  create a separate topic block per problem with equal coverage.

Per topic block:
  topic_label       : 3-5 word label
  query_variants    : 8-10 search phrases (2-4 words) that a REAL member
                      would type. Mix vocabularies:
                      - How members actually talk: "auth denied",
                        "insurance said no", "cant get approved",
                        "waiting for approval"
                      - Clinical terms: "prior authorization denial",
                        "preauthorization refusal"
                      - Process terms: "appeal process", "approval wait"
                      Members search the way they speak — do NOT generate
                      only formal clinical phrasing.
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
You are simulating real health insurance member feedback from NPS surveys.

Generate {HYDE_COUNT} realistic verbatim-style member comments that someone
would write when they had a problem related to the given keyword.

CRITICAL — language style rules:
  - Write like a frustrated or disappointed member, NOT like a clinical report
  - Use informal, conversational language — contractions, run-on sentences,
    and emotional language are all expected and correct
  - Include specific realistic details: wait times, call counts, dollar
    amounts, procedure names, department names
  - Vary the scenario across the {HYDE_COUNT} comments — different aspects,
    different severity, different point in the member journey
  - Do NOT use formal phrasing like "I experienced difficulty obtaining" or
    "the process was suboptimal" — real members never write like that
  - Average length: 2-4 sentences, 30-60 words

GOOD examples of real NPS tone:
  "Been waiting 3 weeks for my auth to go through. Called 5 times, each time
   on hold over an hour and nobody can give me a straight answer. My
   appointment is next week and I'm going to have to cancel."

  "Denied my MRI with no explanation. My doctor says I need it but apparently
   that doesn't matter. Now I have to do an appeal which takes another 30
   days. This is unacceptable."

  "The portal keeps logging me out when I try to check my claim status. Been
   trying for two weeks to find out if my bill was processed."

Return JSON: {{"comments": ["comment1", "comment2", "comment3"]}}
Return ONLY valid JSON. No preamble, no markdown fences.
"""

def get_style_samples(n: int = 4) -> List[str]:
    """
    Pulls n random comments from the index to use as style anchors in the
    HyDE prompt. This grounds HyDE generation in the ACTUAL vocabulary and
    tone of your member base rather than Claude's generic idea of feedback —
    the single biggest lever against generic HyDE output.

    Uses a broad neutral query to fetch a pool, then samples randomly so
    different searches see different style anchors.
    """
    import random
    try:
        result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_text("member feedback complaint"),
            limit=50,
            with_payload=True
        )
        if not result.points:
            return []
        unique_texts = list({hit.payload["original_text"] for hit in result.points})
        return random.sample(unique_texts, min(n, len(unique_texts)))
    except Exception as e:
        print(f"  [WARN] Style sample fetch failed: {e} — proceeding ungrounded")
        return []


def generate_hyde_comments(keyword: str) -> List[str]:
    print(f"\n  [HyDE] Generating grounded comments for '{keyword}' ...")

    style_samples = get_style_samples(n=4)
    if style_samples:
        style_block = "\n\n".join(
            f"Example {i+1}: {s}" for i, s in enumerate(style_samples)
        )
        system_prompt = HYDE_SYSTEM_PROMPT + f"""
STYLE REFERENCE — match the tone, vocabulary, and specificity of these
REAL member comments from this exact member base:

{style_block}
"""
    else:
        system_prompt = HYDE_SYSTEM_PROMPT

    raw      = llm_call(system_prompt, f"Keyword: {keyword}")
    parsed   = parse_json_response(raw)
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

# Full-text index on phrase field — used by Stream B keyword search (Option 2)
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
    related_scenarios so it can be used for TYPE_WEIGHT scoring at query time.
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


# ── Option 2: Stream B Keyword Search ─────────────────────────────────────────

def _stream_b_keyword_search(keyword: str, seen: Dict[int, Dict]) -> None:
    """
    Full-text keyword match against indexed phrases, merged into `seen`.

    For docs already retrieved by Stream A (HyDE vectors): adds KEYWORD_BOOST
    to rrf_score and sets keyword_match=True — they move up in ranking.

    For docs only found here: enters them with rrf_score=KEYWORD_BOOST and
    embedding_score=LOWER_THRESHOLD, landing them in the LLM zone for
    verification rather than auto-including or silently excluding them.

    Uses a should-filter (OR logic) across individual words so "prior auth"
    matches phrases containing either "prior" or "auth".
    """
    words = [w for w in keyword.lower().split() if len(w) >= 2]
    if not words:
        return

    try:
        result, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="phrase",
                        match=models.MatchText(text=w)
                    )
                    for w in words
                ]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        print(f"  [Stream B] Keyword search failed: {e}")
        return

    new_count = 0
    for hit in result:
        doc_id = hit.payload["doc_id"]
        if doc_id in seen:
            seen[doc_id]["rrf_score"]    += KEYWORD_BOOST
            seen[doc_id]["keyword_match"] = True
        else:
            seen[doc_id] = {
                "doc_id"         : doc_id,
                "embedding_score": LOWER_THRESHOLD,
                "rrf_score"      : KEYWORD_BOOST,
                "hit_count"      : 0,
                "keyword_match"  : True,
                "matched_phrase" : hit.payload["phrase"],
                "phrase_type"    : hit.payload["phrase_type"],
                "topic_label"    : hit.payload["topic_label"],
                "all_topics"     : hit.payload["all_topics"],
                "severity"       : hit.payload["severity"],
                "journey_stages" : hit.payload["journey_stages"],
                "original_text"  : hit.payload["original_text"],
            }
            new_count += 1

    print(f"  [Stream B] {len(result)} keyword hits, {new_count} new docs added")


# ── Broad Retrieval ───────────────────────────────────────────────────────────

def retrieve_candidates(keyword: str) -> List[Dict]:
    """
    Stage 1 — HyDE vector search (Stream A) + keyword search (Stream B).

    Stream A — for each HyDE comment, runs a wide Qdrant vector search
    (limit=HYDE_LIMIT). Results are merged with Reciprocal Rank Fusion (RRF)
    weighted by phrase_type (Option 1 + Option 4):
      rrf_contribution = (1 / (60 + rank)) * TYPE_WEIGHT[phrase_type]
    Docs hit by multiple HyDE vectors accumulate rrf_score; the best raw
    cosine similarity is kept as embedding_score for display.

    Stream B — full-text keyword match on indexed phrases (Option 2).
    Boosts docs already in Stream A; adds new keyword-only docs at floor.

    final_score = embedding_score * TYPE_WEIGHT[best_match_phrase_type]
    Used for tier comparison so related_scenario hits are more likely to
    land in the LLM zone than query_variant hits at the same cosine score.

    Returns candidates sorted by rrf_score DESC, no threshold applied.
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
        print(f"  [Stream A] HyDE {i+1} → {len(response.points)} raw hits")

        for rank, hit in enumerate(response.points):
            doc_id           = hit.payload["doc_id"]
            type_weight      = TYPE_WEIGHT.get(hit.payload["phrase_type"], 1.0)
            rrf_contribution = (1 / (60 + rank)) * type_weight

            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id"         : doc_id,
                    "embedding_score": hit.score,
                    "rrf_score"      : rrf_contribution,
                    "hit_count"      : 1,
                    "keyword_match"  : False,
                    "matched_phrase" : hit.payload["phrase"],
                    "phrase_type"    : hit.payload["phrase_type"],
                    "topic_label"    : hit.payload["topic_label"],
                    "all_topics"     : hit.payload["all_topics"],
                    "severity"       : hit.payload["severity"],
                    "journey_stages" : hit.payload["journey_stages"],
                    "original_text"  : hit.payload["original_text"],
                }
            else:
                seen[doc_id]["rrf_score"] += rrf_contribution
                seen[doc_id]["hit_count"] += 1
                # Track the best-scoring phrase for display and final_score base
                if hit.score > seen[doc_id]["embedding_score"]:
                    seen[doc_id]["embedding_score"] = hit.score
                    seen[doc_id]["matched_phrase"]  = hit.payload["phrase"]
                    seen[doc_id]["phrase_type"]     = hit.payload["phrase_type"]
                    seen[doc_id]["topic_label"]     = hit.payload["topic_label"]
                    seen[doc_id]["all_topics"]      = hit.payload["all_topics"]

    # Stream B — keyword match, merges into same `seen` dict
    _stream_b_keyword_search(keyword, seen)

    # final_score = weighted embedding score used for tier comparison (Option 4)
    for doc in seen.values():
        doc["final_score"] = (
            doc["embedding_score"] * TYPE_WEIGHT.get(doc["phrase_type"], 1.0)
        )

    candidates = sorted(seen.values(), key=lambda x: x["rrf_score"], reverse=True)
    print(f"  [Retrieval] {len(candidates)} unique docs after merge\n")
    return candidates


# ── Option 5: Adaptive Threshold Computation ──────────────────────────────────

def _adaptive_thresholds(candidates: List[Dict]) -> Tuple[float, float]:
    """
    Per-query threshold calibration based on the final_score distribution.

    Upper threshold — floored at the score of the 10th-ranked candidate.
    Promotes at least 10 results to high-confidence when the data supports it.
    Clamped to [UPPER_THRESHOLD - 0.10, UPPER_THRESHOLD + 0.05].

    Lower threshold — set to the 75th-percentile score so the LLM zone
    covers the middle quartiles and only the clear tail is auto-excluded.
    Clamped to [LOWER_THRESHOLD - 0.10, LOWER_THRESHOLD + 0.05].

    Falls back to static constants when fewer than 10 candidates exist.
    """
    if len(candidates) < 10:
        return UPPER_THRESHOLD, LOWER_THRESHOLD

    scores = sorted([c["final_score"] for c in candidates], reverse=True)
    n      = len(scores)

    p10_score = scores[min(9, n - 1)]
    upper = float(max(
        min(p10_score, UPPER_THRESHOLD + 0.05),
        UPPER_THRESHOLD - 0.10
    ))

    p75_score = scores[min(int(n * 0.75), n - 1)]
    lower = float(max(
        min(p75_score, LOWER_THRESHOLD + 0.05),
        LOWER_THRESHOLD - 0.10
    ))

    if abs(upper - UPPER_THRESHOLD) > 0.001 or abs(lower - LOWER_THRESHOLD) > 0.001:
        print(f"  [Adaptive] Thresholds adjusted: "
              f"UPPER {UPPER_THRESHOLD:.2f}→{upper:.2f}  "
              f"LOWER {LOWER_THRESHOLD:.2f}→{lower:.2f}")

    return upper, lower


# ── Tiered Threshold Split ────────────────────────────────────────────────────

def tier_candidates(
    candidates: List[Dict]
) -> Tuple[List[Dict], List[Dict], int, float, float]:
    """
    Splits candidates into three tiers using adaptive thresholds (Option 5)
    applied to final_score (embedding_score * phrase_type weight, Option 4).

      Tier 1 (high_confidence) : final_score >= upper
        Auto-included. No LLM call needed.

      Tier 2 (ambiguous)       : lower <= final_score < upper
        Sent to LLM for binary include/exclude.

      Tier 3 (excluded)        : final_score < lower
        Dropped silently.

    Returns (high_confidence, ambiguous, excluded_count, upper, lower).
    The thresholds are returned so render_results can display what was used.
    """
    upper, lower = _adaptive_thresholds(candidates)

    high_confidence: List[Dict] = []
    ambiguous:       List[Dict] = []
    excluded_count               = 0

    for c in candidates:
        score = c["final_score"]
        if score >= upper:
            high_confidence.append(c)
        elif score >= lower:
            ambiguous.append(c)
        else:
            excluded_count += 1

    print(f"  [Tiers] High confidence : {len(high_confidence)} "
          f"(final_score >= {upper:.2f})")
    print(f"  [Tiers] Ambiguous       : {len(ambiguous)} "
          f"({lower:.2f} <= final_score < {upper:.2f})")
    print(f"  [Tiers] Excluded        : {excluded_count} "
          f"(final_score < {lower:.2f})\n")

    return high_confidence, ambiguous, excluded_count, upper, lower


# ── LLM Binary Classification ─────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = """\
You are a health insurance member experience analyst reviewing NPS survey
feedback comments. These comments have been pre-filtered by vector search
as potentially matching the search query.

DOMAIN CONTEXT:
Health insurance NPS comments describe problems across these areas:
prior authorization, claims processing, billing errors, digital portal
issues, customer service quality, pharmacy benefits, specialist referrals,
appeals, and enrollment. Adjacent topics are often relevant — a comment
about specialist referral denial IS relevant to a "prior auth" query
because both involve insurer approval blocking access to care.

DECISION RULES:
  RELEVANT     — comment directly addresses the query topic OR describes a
                 closely related health insurance problem that a member
                 experience analyst would associate with this query.
                 When in doubt and the comment shares the same underlying
                 member problem, lean RELEVANT.

  NOT_RELEVANT — comment is clearly about a completely different topic
                 with no meaningful connection to the query.

CONCRETE EXAMPLES for query "prior auth":
  RELEVANT     : "my authorization was denied"            (direct)
  RELEVANT     : "specialist referral blocked"            (same approval barrier)
  RELEVANT     : "waiting weeks for approval, no update"  (process adjacent)
  NOT_RELEVANT : "my explanation of benefits had the wrong amount"  (billing)
  NOT_RELEVANT : "portal won't load my ID card"           (digital)

Apply the same reasoning pattern to the actual query you receive.

Each comment is shown with the phrase and topic that caused it to be
retrieved. Use this retrieval context to understand WHY the system
considered it a candidate — it is extra signal, not a substitute for
reading the comment text itself.

IMPORTANT: Return one decision per comment using the EXACT doc_id integer
provided in the input. Do not skip any comments.

Return ONLY this JSON structure, no preamble, no markdown fences:
{
  "decisions": [
    {"doc_id": 1, "decision": "RELEVANT"},
    {"doc_id": 2, "decision": "NOT_RELEVANT"}
  ]
}
"""

def classify_batch(batch: List[Dict], keyword: str) -> List[Dict]:
    """
    Classifies one batch of ambiguous candidates as RELEVANT / NOT_RELEVANT.
    Returns the subset decided RELEVANT with llm_verified=True in payload.

    Option 3: each comment is shown with its matched phrase and topic label
    so the LLM sees the same retrieval signal the vector search used, not
    just the raw comment text. This reduces blind misclassification on
    comments that were surfaced via an adjacent concept.
    """
    # Include retrieval context alongside each comment (Option 3)
    formatted = "\n\n".join(
        f"[doc_id={int(c['doc_id'])}]\n"
        f"Retrieval match: '{c['matched_phrase']}' ({c['phrase_type']}) "
        f"| Topic: {c['topic_label']}\n"
        f"Text: {c['original_text']}"
        for c in batch
    )
    user_msg = (
        f'Search query: "{keyword}"\n\n'
        f'Pre-filtered candidate comments:\n\n'
        f'{formatted}'
    )
    raw = llm_call(CLASSIFY_SYSTEM_PROMPT, user_msg)

    print(f"  [Classify DEBUG] Raw LLM response:\n  {raw[:300]}...")

    try:
        parsed = parse_json_response(raw)
        # Cast doc_id to int on both sides — JSON may parse as int or str
        decisions = {
            int(d["doc_id"]): d["decision"]
            for d in parsed.get("decisions", [])
        }
        print(f"  [Classify DEBUG] Parsed {len(decisions)} decisions: "
              f"{decisions}")

    except Exception as e:
        print(f"  [WARN] Classification parse failed: {e}")
        print(f"  [WARN] Raw response was:\n  {raw}")
        # Conservative fallback: include all rather than exclude all
        print(f"  [WARN] Falling back to INCLUDE all {len(batch)} in batch")
        for c in batch:
            c["llm_verified"] = True
        return batch

    verified = []
    for c in batch:
        doc_id   = int(c["doc_id"])
        decision = decisions.get(doc_id, "NOT_RELEVANT")
        print(f"  [Classify DEBUG] doc_id={doc_id} → {decision}")
        if decision == "RELEVANT":
            c["llm_verified"] = True
            verified.append(c)

    print(f"  [Classify] Batch result: "
          f"{len(verified)}/{len(batch)} marked RELEVANT")
    return verified


def classify_ambiguous_parallel(
    ambiguous: List[Dict],
    keyword  : str
) -> List[Dict]:
    """
    Splits ambiguous candidates into batches of CLASSIFY_BATCH and
    runs all classification calls in parallel via ThreadPoolExecutor.
    Returns flat list of LLM-verified relevant candidates, sorted by
    rrf_score DESC (consistent with the retrieval sort order).
    """
    if not ambiguous:
        return []

    batches = [
        ambiguous[i : i + CLASSIFY_BATCH]
        for i in range(0, len(ambiguous), CLASSIFY_BATCH)
    ]

    print(f"  [Classify] {len(ambiguous)} ambiguous → "
          f"{len(batches)} parallel batches ...")

    all_verified: List[Dict] = []

    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = {
            executor.submit(classify_batch, batch, keyword): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            all_verified.extend(future.result())

    all_verified.sort(key=lambda x: x["rrf_score"], reverse=True)

    print(f"  [Classify] {len(all_verified)} / {len(ambiguous)} "
          f"ambiguous passed LLM filter\n")

    return all_verified


# ── Score Distribution Utility ────────────────────────────────────────────────

def print_score_distribution(candidates: List[Dict], keyword: str):
    """
    Prints a histogram of final_score (phrase-type-weighted embedding score)
    across all retrieved candidates. Run on a few real queries to calibrate
    UPPER_THRESHOLD and LOWER_THRESHOLD.

    Look for natural gaps in the distribution — those are where to set
    the thresholds. A gap between 0.70-0.75 suggests UPPER=0.72 etc.
    Tier labels reflect adaptive thresholds computed for this query.
    """
    if not candidates:
        print("  No candidates to analyse.")
        return

    upper, lower = _adaptive_thresholds(candidates)

    buckets: Dict[float, int] = {}
    for c in candidates:
        bucket = round(c["final_score"], 1)
        buckets[bucket] = buckets.get(bucket, 0) + 1

    print(f"\n  Score distribution for query '{keyword}' "
          f"(upper={upper:.2f}, lower={lower:.2f}):")
    for score in sorted(buckets.keys(), reverse=True):
        bar  = "█" * buckets[score]
        tier = (
            "← AUTO-INCLUDE"  if score >= upper  else
            "← LLM ZONE"      if score >= lower  else
            "← AUTO-EXCLUDE"
        )
        print(f"  {score:.1f} | {bar:<30} {buckets[score]:>3}  {tier}")
    print()


# ── Full Search Pipeline ──────────────────────────────────────────────────────

def search(keyword: str) -> Dict:
    """
    Full three-tier search pipeline:

      1. HyDE expansion + Stream B keyword search + broad retrieval
      2. Tier split by final_score with adaptive per-query thresholds
         - High confidence → auto-include
         - Ambiguous       → parallel LLM binary classification
         - Below floor     → auto-exclude
      3. Merge high_confidence + llm_verified into final result set
      4. Both tiers sorted by rrf_score DESC

    Returns dict with keys:
      keyword, high_confidence, llm_verified, excluded_count,
      total_relevant, upper_threshold, lower_threshold
    """
    print(f"\n{'='*62}")
    print(f"  SEARCH: '{keyword}'")
    print(f"{'='*62}\n")

    candidates = retrieve_candidates(keyword)

    # Score distribution for threshold tuning.
    # Comment out once thresholds are calibrated for your data.
    print_score_distribution(candidates, keyword)

    high_confidence, ambiguous, excluded_count, upper, lower = tier_candidates(
        candidates
    )

    llm_verified = classify_ambiguous_parallel(ambiguous, keyword)

    return {
        "keyword"         : keyword,
        "high_confidence" : high_confidence,
        "llm_verified"    : llm_verified,
        "excluded_count"  : excluded_count,
        "total_relevant"  : len(high_confidence) + len(llm_verified),
        "upper_threshold" : upper,
        "lower_threshold" : lower,
    }


# ── Result Rendering ──────────────────────────────────────────────────────────

def render_results(results: Dict):
    """
    Renders final results with tier labels.
    HIGH CONFIDENCE shown first, LLM VERIFIED shown second, both sorted by
    rrf_score DESC. Each result shows rrf_score, raw embedding score,
    final (weighted) score, hit count across HyDE vectors, keyword match
    flag, matched phrase, topic, and journey stages.
    """
    keyword    = results["keyword"]
    hc         = results["high_confidence"]
    lv         = results["llm_verified"]
    excl_count = results["excluded_count"]
    total      = results["total_relevant"]
    upper      = results["upper_threshold"]
    lower      = results["lower_threshold"]
    sep        = "─" * 62

    print(f"\n{'='*62}")
    print(f"  Results for: '{keyword}'")
    print(f"  Total relevant: {total}  "
          f"(HC: {len(hc)}  LLM: {len(lv)}  Excluded: {excl_count})")
    print(f"  Thresholds used: upper={upper:.2f}  lower={lower:.2f}")
    print(f"{'='*62}")

    def _render_row(rank: int, r: Dict):
        kw_flag = " [KW]" if r.get("keyword_match") else ""
        print(f"\n  [{rank}]  RRF: {r['rrf_score']:.4f}  "
              f"Embed: {r['embedding_score']:.4f}  "
              f"Final: {r['final_score']:.4f}  "
              f"Hits: {r['hit_count']}{kw_flag}  "
              f"Doc: {r['doc_id']}  Severity: {r['severity']}")
        print(f"  Topic          : {r['topic_label']}")
        print(f"  Journey stages : {r['journey_stages']}")
        print(f"  Matched phrase : '{r['matched_phrase']}' "
              f"[{r['phrase_type']}]")
        print(f"  Text           : {r['original_text'][:110]}...")

    print(f"\n  ▌HIGH CONFIDENCE  (final_score >= {upper:.2f})")
    print(f"  {sep}")
    if not hc:
        print("  None.")
    for rank, r in enumerate(hc, start=1):
        _render_row(rank, r)

    print(f"\n  ▌LLM VERIFIED  "
          f"({lower:.2f} <= final_score < {upper:.2f}, "
          f"passed Claude filter)")
    print(f"  {sep}")
    if not lv:
        print("  None passed LLM filter.")
    for rank, r in enumerate(lv, start=1):
        _render_row(rank, r)

    print(f"\n  {sep}")
    print(f"  Auto-excluded (final_score < {lower:.2f}): {excl_count} docs")
    llm_batches = (
        (len(lv) // CLASSIFY_BATCH) + (1 if len(lv) % CLASSIFY_BATCH else 0)
    )
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
