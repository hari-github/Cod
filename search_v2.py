# search_v2.py — Member Feedback Search V2
#
# Architecture:
#   Layer 1  — Fuzzy match using healthcare-equivalent terms
#              (true synonyms only — same concept, different expression)
#   Layer 2  — Semantic match using direct comment embeddings
#              + multi-vector query-time expansion
#   Merge    — Reciprocal Rank Fusion across both layers
#   Rerank   — Strict LLM binary classification (default NOT_RELEVANT)
#
# Embedding cache:
#   Comments are embedded once and cached to ./embed_cache/*.npy
#   so subsequent searches load instantly.
#
# Usage:
#   python search_v2.py --csv feedback.csv --text-col comment
#   python search_v2.py --csv feedback.csv --text-col comment --id-col comment_id
#
# Dependencies:
#   pip install pandas numpy rapidfuzz openai google-generativeai

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from provider_databricks import DatabricksProvider
from provider_gemini import GeminiProvider, GEMINI_LLM_MODELS, GEMINI_EMBED_MODEL

Provider = Union[DatabricksProvider, GeminiProvider]

# ── Constants ─────────────────────────────────────────────────────────────────

FUZZY_THRESHOLD  = 70   # token_set_ratio score (0-100); raise to 75 for less noise
TOP_K_SEMANTIC   = 100  # max candidates from semantic layer before merge
TOP_K_MERGE      = 60   # candidates entering LLM rerank after RRF
CLASSIFY_BATCH   = 20   # comments per LLM classification call
RRF_K            = 60   # RRF constant (standard value; higher → flatter merge)

# ── JSON helper ───────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    s = raw.strip()
    for fence in ["```json", "```"]:
        if s.startswith(fence):
            s = s[len(fence):]
    if s.endswith("```"):
        s = s[:-3]
    return json.loads(s.strip())

# ═════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Fuzzy match with healthcare-equivalent terms
# ═════════════════════════════════════════════════════════════════════════════

_EQUIV_SYSTEM = """\
You are a US health insurance terminology expert.

Given a search query, generate ONLY terms that mean EXACTLY the same thing
in US health insurance / healthcare.

WHAT TO INCLUDE:
  - Alternative spellings of the same term
  - Common abbreviations and their expansions
  - Informal member-facing shorthand for the same process
  - Synonyms that a member or admin would treat as interchangeable

WHAT TO EXCLUDE — strictly:
  - Related but distinct processes (different steps in the member journey)
  - Downstream effects of the concept
  - Broader categories the concept belongs to
  - Anything a clinician or analyst would call "related but different"

EXAMPLES:
  "prior auth"   → ["prior authorization", "preauthorization", "preauth",
                     "PA", "auth request", "authorization request",
                     "pre-approval", "prior auth"]

  "copay"        → ["co-pay", "copayment", "co-payment",
                     "member cost share", "patient cost share"]

  "EOB"          → ["explanation of benefits", "benefit statement",
                     "remittance advice", "EOB form"]

  "deductible"   → ["annual deductible", "yearly deductible",
                     "deductible amount", "ded"]

  "formulary"    → ["drug formulary", "prescription formulary",
                     "preferred drug list", "PDL", "covered drug list"]

NEGATIVE EXAMPLES — do NOT generate:
  "prior auth"   ✗ "claim denial"        (different process — happens after service)
  "prior auth"   ✗ "specialist referral" (different type of approval)
  "prior auth"   ✗ "step therapy"        (different UM tool)
  "copay"        ✗ "deductible"          (different cost-sharing mechanism)
  "copay"        ✗ "out of pocket max"   (different concept)

Return ONLY valid JSON — no preamble, no markdown:
{"equivalents": ["term1", "term2", ...]}
"""


def generate_equivalents(provider: Provider, query: str) -> List[str]:
    """Returns healthcare-equivalent terms for the query. Always includes the original."""
    raw = provider.llm_call(_EQUIV_SYSTEM, f'Query: "{query}"')
    try:
        terms = _parse_json(raw).get("equivalents", [])
    except Exception:
        terms = []

    # Deduplicate, preserve order, always keep the original query first
    seen: set = set()
    result: List[str] = []
    for t in [query] + terms:
        key = t.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(t)
    return result


def fuzzy_search(
    equivalent_terms: List[str],
    documents: List[Dict],
    threshold: int = FUZZY_THRESHOLD,
) -> List[Dict]:
    """
    Scores every comment against every equivalent term using token_set_ratio.
    Returns the best-scoring match per comment, filtered by threshold.
    token_set_ratio handles word order and partial matches well for short queries
    against longer comment texts.
    """
    best: Dict[int, Dict] = {}
    for term in equivalent_terms:
        t_lower = term.lower()
        for doc in documents:
            score = fuzz.token_set_ratio(t_lower, doc["text"].lower())
            if score >= threshold:
                doc_id = doc["doc_id"]
                if doc_id not in best or score > best[doc_id]["fuzzy_score"]:
                    best[doc_id] = {
                        **doc,
                        "fuzzy_score":  score / 100.0,
                        "matched_term": term,
                        "layers":       {"fuzzy"},
                    }
    return sorted(best.values(), key=lambda x: x["fuzzy_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Semantic search with query-time expansion
# ═════════════════════════════════════════════════════════════════════════════

_SEMANTIC_SYSTEM = """\
You are a health insurance member experience search specialist.

Generate diverse phrases for embedding-based retrieval against member NPS comments.
Each phrase represents a different way a MEMBER might DESCRIBE the same core
problem — not adjacent problems, but different expressions of the SAME situation.

RULES:
  - Write from the member's perspective, in the member's voice
  - Cover different severity levels and journey stages of the SAME issue
  - Use natural, conversational language — the way frustrated members write
  - 6-8 phrases, each 8-20 words
  - Do NOT generate adjacent topics or related-but-different problems

GOOD for "prior auth":
  "my insurance authorization was denied and I can't get my treatment"
  "been waiting weeks for prior authorization approval with no update"
  "insurance keeps blocking my procedure I need pre-approval"
  "prior auth still pending after multiple calls to member services"
  "doctor submitted authorization request but insurance hasn't responded"
  "preauthorization denied without a clear explanation from the insurance company"

BAD for "prior auth" — do NOT generate:
  "pharmacy wouldn't fill my prescription"             ← pharmacy, different topic
  "my claim was rejected after the procedure"          ← claims, different topic
  "specialist referral was not covered by my plan"     ← referral, different topic

Return ONLY valid JSON — no preamble, no markdown:
{"variants": ["phrase1", "phrase2", ...]}
"""


def generate_semantic_variants(provider: Provider, query: str) -> List[str]:
    """Returns phrasing variants of the query for multi-vector embedding search."""
    raw = provider.llm_call(_SEMANTIC_SYSTEM, f'Search query: "{query}"')
    try:
        variants = _parse_json(raw).get("variants", [])
    except Exception:
        variants = []
    return variants if variants else [query]


def semantic_search(
    query_variants: List[str],
    provider:       Provider,
    doc_matrix:     np.ndarray,
    documents:      List[Dict],
    top_k:          int = TOP_K_SEMANTIC,
) -> List[Dict]:
    """
    Embeds each variant and computes cosine similarity against the doc matrix.
    Merges across variants with RRF — documents hit by multiple query vectors
    accumulate score, surfacing consistently relevant comments.
    """
    rrf_scores:   Dict[int, float] = {}
    best_cosines: Dict[int, float] = {}
    norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8
    normed_matrix = doc_matrix / norms  # pre-normalise once

    for variant in query_variants:
        qvec  = np.array(provider.embed_text(variant, task_type="retrieval_query"), dtype=np.float32)
        qnorm = np.linalg.norm(qvec) + 1e-8
        sims  = normed_matrix @ (qvec / qnorm)  # cosine similarity for each doc
        ranked = np.argsort(-sims)

        for rank, idx in enumerate(ranked[:top_k]):
            doc_id = documents[int(idx)]["doc_id"]
            rrf_scores[doc_id]    = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            best_cosines[doc_id]  = max(best_cosines.get(doc_id, -1.0), float(sims[idx]))

    results = []
    for doc_id, rrf in sorted(rrf_scores.items(), key=lambda x: -x[1]):
        doc = next(d for d in documents if d["doc_id"] == doc_id)
        results.append({
            **doc,
            "semantic_rrf":  rrf,
            "semantic_score": best_cosines[doc_id],
            "layers":        {"semantic"},
        })
    return results[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# MERGE — Reciprocal Rank Fusion
# ═════════════════════════════════════════════════════════════════════════════

def rrf_merge(
    fuzzy_results:    List[Dict],
    semantic_results: List[Dict],
    top_k:            int = TOP_K_MERGE,
) -> List[Dict]:
    """
    Combines both retrieval layers via RRF.
    Documents appearing in both layers accumulate contributions from each rank,
    reliably surfacing comments that strong signal in either or both streams.
    """
    scores:  Dict[int, float] = {}
    merged:  Dict[int, Dict]  = {}

    for rank, r in enumerate(fuzzy_results):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        merged[doc_id] = dict(r)  # fuzzy fields: fuzzy_score, matched_term

    for rank, r in enumerate(semantic_results):
        doc_id = r["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        if doc_id in merged:
            merged[doc_id]["layers"].add("semantic")
            merged[doc_id]["semantic_rrf"]   = r.get("semantic_rrf", 0.0)
            merged[doc_id]["semantic_score"] = r.get("semantic_score", 0.0)
        else:
            merged[doc_id] = dict(r)

    for doc_id, r in merged.items():
        r["rrf_score"] = scores[doc_id]

    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# RERANK — Strict LLM binary classification
# ═════════════════════════════════════════════════════════════════════════════

_RERANK_SYSTEM = """\
You are a health insurance member experience analyst.

Decide which comments are DIRECTLY relevant to the search query.

DEFAULT: NOT_RELEVANT. Only override to RELEVANT when you see clear, direct
evidence in the comment text itself.

RELEVANT criteria (ALL must apply):
  1. The comment describes the SAME process, problem, or concept as the query
  2. A member analyst would file this comment under the query topic
  3. The core issue — not just a passing mention — matches the query

NOT_RELEVANT: comment is about a different topic even if related or adjacent.
A passing mention of the query term is NOT enough — the main complaint must
be about the query topic.

Apply the same logic to whatever query you receive.

Return ONLY this JSON — no preamble, no markdown:
{"decisions": [{"doc_id": 1, "decision": "RELEVANT"}, ...]}
"""


def _classify_batch(batch: List[Dict], query: str, provider: Provider) -> List[Dict]:
    formatted = "\n\n".join(
        f"[doc_id={d['doc_id']}]\n{d['text']}"
        for d in batch
    )
    raw = provider.llm_call(
        _RERANK_SYSTEM,
        f'Search query: "{query}"\n\nComments to classify:\n\n{formatted}',
    )
    try:
        decisions = {
            int(d["doc_id"]): d["decision"]
            for d in _parse_json(raw).get("decisions", [])
        }
    except Exception:
        return batch  # conservative fallback: include all on parse error

    return [c for c in batch if decisions.get(int(c["doc_id"]), "NOT_RELEVANT") == "RELEVANT"]


def llm_rerank(candidates: List[Dict], query: str, provider: Provider) -> List[Dict]:
    """Parallel batch classification. Returns verified RELEVANT results sorted by RRF."""
    if not candidates:
        return []
    batches = [candidates[i : i + CLASSIFY_BATCH] for i in range(0, len(candidates), CLASSIFY_BATCH)]
    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=max(len(batches), 1)) as ex:
        futures = [ex.submit(_classify_batch, b, query, provider) for b in batches]
        for f in as_completed(futures):
            results.extend(f.result())
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# ═════════════════════════════════════════════════════════════════════════════

def build_or_load_embeddings(
    documents:  List[Dict],
    provider:   Provider,
    cache_path: Path,
) -> np.ndarray:
    """
    Loads cached embeddings if available; otherwise embeds all comments and
    saves to disk.  Cache filename encodes provider + model so switching
    providers automatically rebuilds.
    """
    if cache_path.exists():
        print(f"  Loading embedding cache: {cache_path.name}")
        return np.load(str(cache_path)).astype(np.float32)

    print(f"  Embedding {len(documents)} comments (runs once, then cached) ...")
    vecs: List[List[float]] = []
    for i, doc in enumerate(documents):
        vec = provider.embed_text(doc["text"], task_type="retrieval_document")
        vecs.append(vec)
        if (i + 1) % 25 == 0 or (i + 1) == len(documents):
            print(f"    {i + 1}/{len(documents)}")
    matrix = np.array(vecs, dtype=np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), matrix)
    print(f"  Saved to {cache_path}")
    return matrix


# ═════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def search(
    query:           str,
    documents:       List[Dict],
    doc_matrix:      np.ndarray,
    provider:        Provider,
    fuzzy_threshold: int  = FUZZY_THRESHOLD,
    top_k_merge:     int  = TOP_K_MERGE,
    verbose:         bool = True,
) -> List[Dict]:
    if verbose:
        print(f"\n{'='*62}")
        print(f"  Query: '{query}'")
        print(f"{'='*62}")

    # Step 1 — Healthcare equivalents (fuzzy layer)
    if verbose: print("\n  [1/4] Healthcare equivalent terms ...")
    equivalents = generate_equivalents(provider, query)
    if verbose:
        for t in equivalents:
            print(f"        · {t}")

    # Step 2 — Semantic variants (dense layer)
    if verbose: print("\n  [2/4] Semantic search variants ...")
    variants = generate_semantic_variants(provider, query)
    if verbose:
        for v in variants:
            print(f"        · {v}")

    # Step 3 — Run both retrieval layers
    if verbose: print("\n  [3/4] Running retrieval layers ...")
    fuzzy_res    = fuzzy_search(equivalents, documents, fuzzy_threshold)
    semantic_res = semantic_search(variants, provider, doc_matrix, documents)
    merged       = rrf_merge(fuzzy_res, semantic_res, top_k_merge)
    if verbose:
        fuzzy_ids    = {r["doc_id"] for r in fuzzy_res}
        semantic_ids = {r["doc_id"] for r in semantic_res}
        overlap      = fuzzy_ids & semantic_ids
        print(f"        Fuzzy    : {len(fuzzy_res)} hits")
        print(f"        Semantic : {len(semantic_res)} hits")
        print(f"        Overlap  : {len(overlap)} docs found by both layers")
        print(f"        Merged   : {len(merged)} candidates into LLM rerank")

    # Step 4 — LLM rerank
    if verbose: print(f"\n  [4/4] LLM rerank ({len(merged)} candidates) ...")
    final = llm_rerank(merged, query, provider)
    if verbose:
        only_fuzzy    = sum(1 for r in final if r["layers"] == {"fuzzy"})
        only_semantic = sum(1 for r in final if r["layers"] == {"semantic"})
        both          = sum(1 for r in final if "fuzzy" in r["layers"] and "semantic" in r["layers"])
        print(f"        Relevant : {len(final)}")
        print(f"          Fuzzy-only: {only_fuzzy}  |  Semantic-only: {only_semantic}  |  Both: {both}")
        print(f"{'='*62}")

    return final


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def render_results(results: List[Dict], query: str):
    sep = "─" * 62
    if not results:
        print(f"\n  {sep}")
        print(f"  No relevant comments found for: '{query}'")
        print(f"  {sep}")
        return

    print(f"\n  {len(results)} relevant comment(s) for: '{query}'")
    print(f"  {sep}")

    for i, r in enumerate(results, 1):
        layers  = " + ".join(sorted(r.get("layers", set())))
        rrf     = r.get("rrf_score", 0.0)
        fscore  = r.get("fuzzy_score", 0.0)
        sscore  = r.get("semantic_score", 0.0)
        matched = r.get("matched_term", "")

        scores_str = f"RRF {rrf:.4f}"
        if fscore:  scores_str += f"  Fuzzy {fscore:.0%}"
        if sscore:  scores_str += f"  Cosine {sscore:.3f}"

        print(f"\n  [{i:>3}]  [{layers}]  {scores_str}")
        if matched:
            print(f"         Matched term : '{matched}'")
        text = r["text"]
        print(f"         {text[:130]}{'...' if len(text) > 130 else ''}")

    print(f"\n  {sep}")


# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def select_provider() -> Provider:
    print("\nSelect AI provider:")
    print("  1  Databricks  (OpenAI-compatible endpoint)")
    print("  2  Gemini      (Google AI — 2 s delay between calls)")
    choice = input("Provider (1/2) [default 1]: ").strip() or "1"

    if choice == "2":
        api_key     = os.environ.get("GEMINI_API_KEY")     or input("Gemini API key: ").strip()
        llm_model   = (os.environ.get("GEMINI_LLM_MODEL")
                       or input(f"LLM model [{GEMINI_LLM_MODELS[0]}]: ").strip()
                       or GEMINI_LLM_MODELS[0])
        embed_model = (os.environ.get("GEMINI_EMBED_MODEL")
                       or input(f"Embed model [{GEMINI_EMBED_MODEL}]: ").strip()
                       or GEMINI_EMBED_MODEL)
        provider = GeminiProvider(api_key, llm_model, embed_model)
    else:
        base_url    = os.environ.get("DATABRICKS_BASE_URL") or input("Databricks Base URL : ").strip()
        token       = os.environ.get("DATABRICKS_TOKEN")    or input("Databricks Token    : ").strip()
        llm_model   = os.environ.get("LLM_MODEL")           or input("LLM model name      : ").strip()
        embed_model = os.environ.get("EMBED_MODEL")         or input("Embedding model name: ").strip()
        provider = DatabricksProvider(base_url, token, llm_model, embed_model)

    print(f"\n  Provider    : {provider.name}")
    print(f"  LLM model   : {provider.llm_model}")
    print(f"  Embed model : {provider.embed_model}")
    return provider


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Member Feedback Search V2 — query-time expansion + direct embeddings"
    )
    parser.add_argument("--csv",             required=True,           help="Input CSV file path")
    parser.add_argument("--text-col",        default="comment",       help="Column containing comment text")
    parser.add_argument("--id-col",          default=None,            help="Column containing document IDs")
    parser.add_argument("--cache-dir",       default="./embed_cache", help="Directory for embedding cache files")
    parser.add_argument("--fuzzy-threshold", type=int, default=FUZZY_THRESHOLD,
                        help=f"Fuzzy match threshold 0-100 (default {FUZZY_THRESHOLD}). "
                             f"Raise to reduce noise, lower to catch more.")
    parser.add_argument("--top-k",          type=int, default=TOP_K_MERGE,
                        help=f"Max candidates sent to LLM rerank (default {TOP_K_MERGE})")
    parser.add_argument("--query",          default=None,
                        help="Run a single query and exit (non-interactive mode)")
    args = parser.parse_args()

    # Load CSV
    print(f"\nLoading '{args.csv}' ...")
    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(
            f"Column '{args.text_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df.dropna(subset=[args.text_col]).reset_index(drop=True)

    id_col = args.id_col if args.id_col and args.id_col in df.columns else None
    documents: List[Dict] = [
        {
            "doc_id": int(df[id_col].iloc[i]) if id_col else i + 1,
            "text":   str(df[args.text_col].iloc[i]),
        }
        for i in range(len(df))
    ]
    print(f"  {len(documents)} comments loaded")

    # Provider
    provider = select_provider()

    # Embedding cache — filename encodes provider + model to avoid stale cache
    cache_dir  = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_tag  = provider.embed_model.replace("/", "_").replace(".", "_").replace("-", "_")
    cache_path = cache_dir / f"{provider.name.lower()}_{model_tag}.npy"
    doc_matrix = build_or_load_embeddings(documents, provider, cache_path)

    print(f"\n  Ready — {len(documents)} comments indexed")
    print(f"  Fuzzy threshold : {args.fuzzy_threshold}")
    print(f"  Top-K to rerank : {args.top_k}")
    print(f"  Overlap (both layers) is reported per query so you can tune threshold.\n")

    if args.query:
        # Single-query mode (useful for scripting / testing)
        results = search(
            query=args.query, documents=documents, doc_matrix=doc_matrix,
            provider=provider, fuzzy_threshold=args.fuzzy_threshold,
            top_k_merge=args.top_k,
        )
        render_results(results, args.query)
        return

    # Interactive loop
    print("Type a search query, or 'quit' to exit.\n")
    while True:
        try:
            raw_query = input("Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not raw_query or raw_query.lower() in ("quit", "exit", "q"):
            break
        results = search(
            query=raw_query, documents=documents, doc_matrix=doc_matrix,
            provider=provider, fuzzy_threshold=args.fuzzy_threshold,
            top_k_merge=args.top_k,
        )
        render_results(results, raw_query)


if __name__ == "__main__":
    main()
