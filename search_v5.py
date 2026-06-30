# search_v5.py — Pure-fuzzy member feedback search (thesaurus + LLM expansion)
#
# Flow per query (see also build_thesaurus.py / fuzzy_eval.py):
#   1. Build the CONTEXT SET = frozen thesaurus terms  +  live LLM expansion
#      (both feed the same fuzzy matcher; LLM adds vocabulary the static
#       thesaurus may not cover for this exact phrasing)
#   2. Fuzzy match the context set against every comment  →  candidate pool
#      (partial_ratio + word-boundary regex for short tokens; threshold 80)
#   3. OPTIONAL LLM context filter — asked INLINE at runtime (y/N), not a flag.
#      Removes surface matches that are negated / wrong-sense / off-topic.
#
# Cost note: the only per-document LLM work is the optional filter over the
# small candidate pool (~16 docs/query here). Expansion is a fixed handful of
# calls independent of corpus size.
#
# Usage (Databricks Foundation Model serving):
#   set DATABRICKS_BASE_URL=https://<workspace>.cloud.databricks.com/serving-endpoints
#   set DATABRICKS_TOKEN=dapi...
#   python search_v5.py --csv sample_comments.csv
#
# Deps: pandas, rapidfuzz, openai

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

# ── Tunables (from fuzzy_eval.py calibration) ───────────────────────────────
FUZZY_THRESHOLD     = 80    # partial_ratio cutoff to enter the candidate pool
SHORT_TOKEN_MAXLEN  = 3     # short tokens (PA, ER, rep) use word-boundary regex
THESAURUS_MATCH_MIN = 85    # token_set_ratio to attach a thesaurus topic to a query
CLASSIFY_BATCH      = 15    # comments per LLM filter call
THESAURUS_PATH      = "thesaurus.json"

# ── JSON helper (tolerant of fences / preamble) ─────────────────────────────
def parse_json(raw: str, anchor: str) -> dict:
    s = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\"" + re.escape(anchor) + r"\".*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — CONTEXT SET  (thesaurus  +  LLM expansion)
# ═════════════════════════════════════════════════════════════════════════════
_EXPAND_SYSTEM = """\
You are a US health insurance terminology expert helping a keyword search.
Given a search phrase, return the words and SHORT phrases members and admins use
to refer to the SAME concept — so a fuzzy text matcher can find their comments.

INCLUDE: alternative spellings, abbreviations + expansions, informal member
shorthand, and common phrasings for the same issue.
EXCLUDE: related-but-different processes, downstream effects, broader categories.

Each term 1-4 words, lowercase unless an acronym.
Return ONLY JSON: {"terms": ["term1", "term2", ...]}
"""

def llm_expansion(provider, query: str) -> List[str]:
    """Live LLM expansion of the query into matchable vocabulary."""
    try:
        raw = provider.llm_call(_EXPAND_SYSTEM, f'Search phrase: "{query}"')
        return [t for t in parse_json(raw, "terms").get("terms", [])
                if isinstance(t, str) and t.strip()]
    except Exception as e:
        print(f"   ! LLM expansion failed ({e}); using thesaurus only", file=sys.stderr)
        return []

def thesaurus_expansion(query: str, thesaurus: Dict[str, List[str]]) -> List[str]:
    """Pull synonym lists from any thesaurus topic this query matches."""
    qn = default_process(query) or query.lower()
    out: List[str] = []
    for topic, terms in thesaurus.items():
        if topic.startswith("_"):
            continue
        candidates = [topic] + terms
        best = max(fuzz.token_set_ratio(qn, default_process(c) or c.lower())
                   for c in candidates)
        if best >= THESAURUS_MATCH_MIN:
            out.extend([topic] + terms)
    return out

def build_context_set(query, thesaurus, provider) -> Tuple[List[str], Dict[str, int]]:
    """Merge query + thesaurus + LLM expansion, dedup, preserve order."""
    th  = thesaurus_expansion(query, thesaurus)
    llm = llm_expansion(provider, query)
    seen, terms = set(), []
    for t in [query] + th + llm:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            terms.append(t)
    return terms, {"thesaurus": len(th), "llm": len(llm), "total": len(terms)}

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — FUZZY MATCH  →  candidate pool   (CPU only, no LLM)
# ═════════════════════════════════════════════════════════════════════════════
def term_score(term: str, text_proc: str) -> float:
    t = default_process(term) or term.lower()
    if len(t) <= SHORT_TOKEN_MAXLEN and " " not in t:
        return 100.0 if re.search(rf"\b{re.escape(t)}\b", text_proc) else 0.0
    return float(fuzz.partial_ratio(t, text_proc))

def fuzzy_pool(terms, docs, threshold) -> List[Dict]:
    pool = []
    for d in docs:
        best, best_term = 0.0, None
        for t in terms:
            s = term_score(t, d["proc"])
            if s > best:
                best, best_term = s, t
        if best >= threshold:
            pool.append({**d, "score": best, "matched": best_term})
    pool.sort(key=lambda x: x["score"], reverse=True)
    return pool

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — OPTIONAL LLM CONTEXT FILTER  (precision; inline opt-in)
# ═════════════════════════════════════════════════════════════════════════════
_FILTER_SYSTEM = """\
You are a STRICT relevance judge for health-insurance member feedback search.
Given a SEARCH TOPIC and numbered member comments, return the numbers of ONLY
the comments genuinely ABOUT that topic.

A comment is NOT relevant if it merely contains a matching word but is about
something else, or if it explicitly says the topic did NOT apply. When unsure,
EXCLUDE.

Return ONLY JSON: {"relevant": [numbers]}
"""

def llm_filter(provider, query: str, pool: List[Dict]) -> List[Dict]:
    kept: List[Dict] = []
    for i in range(0, len(pool), CLASSIFY_BATCH):
        batch = pool[i:i + CLASSIFY_BATCH]
        listing = "\n".join(f"{n+1}. {d['text']}" for n, d in enumerate(batch))
        user = f'SEARCH TOPIC: "{query}"\n\nCOMMENTS:\n{listing}'
        try:
            raw = provider.llm_call(_FILTER_SYSTEM, user)
            rel = set(parse_json(raw, "relevant").get("relevant", []))
        except Exception as e:
            print(f"   ! filter batch failed ({e}); keeping batch as-is", file=sys.stderr)
            rel = set(range(1, len(batch) + 1))
        kept.extend(d for n, d in enumerate(batch, start=1) if n in rel)
    return kept

# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER + RENDER
# ═════════════════════════════════════════════════════════════════════════════
def make_provider(model: str):
    from provider_databricks import DatabricksProvider
    base_url = os.environ.get("DATABRICKS_BASE_URL")
    token    = os.environ.get("DATABRICKS_TOKEN")
    if not base_url or not token:
        sys.exit("Set DATABRICKS_BASE_URL and DATABRICKS_TOKEN and retry.")
    embed_model = os.environ.get("DATABRICKS_EMBED_MODEL", "databricks-bge-large-en")
    return DatabricksProvider(base_url, token, llm_model=model, embed_model=embed_model)

def render(results: List[Dict], query: str, filtered: bool):
    tag = "LLM-filtered" if filtered else "fuzzy candidate pool"
    print(f"\n  {len(results)} results for \"{query}\"  ({tag})\n  " + "─" * 60)
    for r in results:
        print(f"  [{r['score']:5.1f}] #{r['doc_id']}  (via '{r['matched']}')")
        print(f"          {r['text']}")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN — interactive; the filter choice is asked INLINE, not via argparse
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Fuzzy member-feedback search (thesaurus + LLM expansion)")
    ap.add_argument("--csv",       default="sample_comments.csv")
    ap.add_argument("--text-col",  default="comment")
    ap.add_argument("--id-col",    default="comment_id")
    ap.add_argument("--model",     default="databricks-meta-llama-3-3-70b-instruct",
                    help="Databricks serving endpoint for expansion + filter")
    ap.add_argument("--threshold", type=int, default=FUZZY_THRESHOLD)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col])
    docs = [{
        "doc_id": int(df[args.id_col].iloc[i]) if args.id_col in df.columns else i + 1,
        "text":   str(df[args.text_col].iloc[i]),
        "proc":   default_process(str(df[args.text_col].iloc[i])) or str(df[args.text_col].iloc[i]).lower(),
    } for i in range(len(df))]

    thesaurus = json.loads(Path(THESAURUS_PATH).read_text(encoding="utf-8")) \
        if Path(THESAURUS_PATH).exists() else {}
    provider = make_provider(args.model)

    print(f"\n  {len(docs)} comments · thesaurus topics: "
          f"{len([k for k in thesaurus if not k.startswith('_')])} · model: {args.model}")
    print("  Type a search phrase, or 'quit'.\n")

    while True:
        try:
            query = input("Search > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye."); break
        if not query or query.lower() in ("quit", "exit"):
            print("Bye."); break

        # STEP 1
        terms, stats = build_context_set(query, thesaurus, provider)
        print(f"   context set: {stats['total']} terms "
              f"(thesaurus {stats['thesaurus']} + LLM {stats['llm']})")

        # STEP 2
        pool = fuzzy_pool(terms, docs, args.threshold)
        print(f"   fuzzy candidate pool: {len(pool)} comments (≥ {args.threshold})")

        # STEP 3 — inline opt-in
        results, filtered = pool, False
        if pool:
            choice = input("   Apply LLM context filter? [y/N]: ").strip().lower()
            if choice in ("y", "yes"):
                results = llm_filter(provider, query, pool)
                filtered = True
                print(f"   filter kept {len(results)}/{len(pool)}")

        render(results, query, filtered)


if __name__ == "__main__":
    main()
