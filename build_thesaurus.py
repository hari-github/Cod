# build_thesaurus.py — Corpus-grounded synonym thesaurus builder (monthly job)
#
# Builds thesaurus.json from the ACTUAL member comments so the synonym sets
# reflect how members really phrase things, not hand-guesses. Two LLM passes,
# per topic:
#
#   Pass 1  EXPAND   (generator model) — read a sample of the topic's real
#                     comments, extract the distinct surface forms / spellings /
#                     abbreviations / member shorthand for the topic's concept.
#   Pass 2  VALIDATE (reverse-validation, ideally a DIFFERENT model) — keep only
#                     terms that are TRUE synonyms of the concept; drop
#                     related-but-different / downstream / broader terms.
#
# Then an optional NO-LLM discriminativeness check reports how cleanly each term
# separates its own topic from the rest (purely from the corpus).
#
# Safety for the recurring run: backs up the existing thesaurus and prints a
# diff (added / removed terms per topic) so each monthly run is auditable.
#
# Usage (Databricks Foundation Model serving):
#   set DATABRICKS_BASE_URL=https://<workspace>.cloud.databricks.com/serving-endpoints
#   set DATABRICKS_TOKEN=dapi...
#   python build_thesaurus.py --csv sample_comments.csv --dry-run    # preview only
#   python build_thesaurus.py --csv sample_comments.csv              # write it
#   python build_thesaurus.py --gen-model databricks-meta-llama-3-3-70b-instruct \
#                             --validate-model databricks-mixtral-8x7b-instruct
#
# --gen-model / --validate-model are serving-endpoint names; set them to whatever
# Foundation Models your workspace exposes. Schedule monthly via Task Scheduler / cron.
#
# Deps: pandas, rapidfuzz, openai

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

# ── Tunables ────────────────────────────────────────────────────────────────
MAX_COMMENTS_PER_TOPIC = 60      # cap the sample sent to the LLM (token/cost control)
SAMPLE_SEED            = 42      # deterministic sampling so monthly diffs are meaningful
SHORT_TOKEN_MAXLEN     = 3       # short tokens (PA, ER, rep) use word-boundary regex
DISCRIM_FUZZY_THRESH   = 80      # partial_ratio cutoff for the discriminativeness check

# ── JSON helper (tolerant of markdown fences / preamble) ─────────────────────
def parse_json(raw: str, anchor: str) -> dict:
    s = raw.strip()
    # strip ```json ... ``` fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        # last resort: grab the first {...} block containing the anchor key
        m = re.search(r"\{[^{}]*\"" + re.escape(anchor) + r"\"[^{}]*\}", s, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

# ═════════════════════════════════════════════════════════════════════════════
# PASS 1 — EXPAND (grounded in real comments)
# ═════════════════════════════════════════════════════════════════════════════
_EXPAND_SYSTEM = """\
You are a US health insurance terminology expert building a SEARCH synonym
thesaurus. You are given a TOPIC and a sample of real member feedback comments
tagged with that topic.

Extract the distinct words and short phrases members and admins use to refer to
the topic's CORE concept — grounded in how these comments actually phrase it.

INCLUDE:
  - the canonical term for the concept
  - alternative spellings of the same term
  - abbreviations and their expansions (e.g. "PA" / "prior auth")
  - informal member-facing shorthand for the same thing

EXCLUDE strictly:
  - related-but-different processes (different steps in the journey)
  - downstream effects or causes of the concept
  - broader categories the concept belongs to
  - generic words that appear but don't denote the concept

Each term must be 1-4 words. Lowercase unless it is an acronym.
Return ONLY JSON, no preamble, no markdown:
{"terms": ["term1", "term2", ...]}
"""

def expand_terms(provider, topic: str, comments: List[str]) -> List[str]:
    sample = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    user = f'TOPIC: "{topic}"\n\nSAMPLE COMMENTS:\n{sample}'
    raw = provider.llm_call(_EXPAND_SYSTEM, user)
    try:
        terms = parse_json(raw, "terms").get("terms", [])
    except Exception as e:
        print(f"   ! expand parse failed for '{topic}': {e}", file=sys.stderr)
        terms = []
    return [t for t in terms if isinstance(t, str) and t.strip()]

# ═════════════════════════════════════════════════════════════════════════════
# PASS 2 — REVERSE VALIDATION (independent model)
# ═════════════════════════════════════════════════════════════════════════════
_VALIDATE_SYSTEM = """\
You are a STRICT validator for a US health-insurance search thesaurus.

Given a CONCEPT and a list of CANDIDATE terms, return only the candidates that
are TRUE synonyms or alternative surface forms of the concept — interchangeable
ways to refer to the SAME thing (spellings, abbreviations, expansions, member
shorthand).

REJECT a candidate if it is any of:
  - a related-but-different process
  - a downstream effect or a cause of the concept
  - a broader category the concept belongs to
  - a term that merely co-occurs with the concept

Return ONLY the exact candidate strings you keep, as JSON, no preamble:
{"valid": ["term1", "term2", ...]}
"""

def validate_terms(provider, concept: str, candidates: List[str]) -> List[str]:
    if not candidates:
        return []
    user = f'CONCEPT: "{concept}"\n\nCANDIDATES: {json.dumps(candidates)}'
    raw = provider.llm_call(_VALIDATE_SYSTEM, user)
    try:
        valid = parse_json(raw, "valid").get("valid", [])
    except Exception as e:
        print(f"   ! validate parse failed for '{concept}': {e}", file=sys.stderr)
        return candidates  # fail open — keep generator output rather than dropping all
    keep = {v.lower().strip() for v in valid if isinstance(v, str)}
    # preserve original candidate casing/order, keep only validated ones
    return [c for c in candidates if c.lower().strip() in keep]

# ═════════════════════════════════════════════════════════════════════════════
# NO-LLM discriminativeness check (purely from the corpus)
# ═════════════════════════════════════════════════════════════════════════════
def _term_hits(term: str, texts_proc: List[str]) -> int:
    t = default_process(term) or term.lower()
    if len(t) <= SHORT_TOKEN_MAXLEN and " " not in t:
        pat = re.compile(rf"\b{re.escape(t)}\b")
        return sum(1 for x in texts_proc if pat.search(x))
    return sum(1 for x in texts_proc if fuzz.partial_ratio(t, x) >= DISCRIM_FUZZY_THRESH)

def discriminativeness(term, in_topic_proc, off_topic_proc) -> Dict:
    """How cleanly a term picks out its own topic vs the rest of the corpus."""
    in_rate  = _term_hits(term, in_topic_proc)  / max(1, len(in_topic_proc))
    off_rate = _term_hits(term, off_topic_proc) / max(1, len(off_topic_proc))
    return {"term": term, "in_rate": round(in_rate, 3), "off_rate": round(off_rate, 3)}

# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER
# ═════════════════════════════════════════════════════════════════════════════
def make_provider(model: str):
    """Build a DatabricksProvider pointed at a Foundation Model serving endpoint.

    Needs two env vars:
      DATABRICKS_BASE_URL  e.g. https://<workspace>.cloud.databricks.com/serving-endpoints
      DATABRICKS_TOKEN     a personal access token / service-principal token
    `model` is the serving-endpoint name (e.g. databricks-meta-llama-3-3-70b-instruct).
    The embed endpoint is unused here (thesaurus build is LLM-only) but the provider
    constructor requires one, so we pass a harmless default.
    """
    from provider_databricks import DatabricksProvider
    base_url = os.environ.get("DATABRICKS_BASE_URL")
    token    = os.environ.get("DATABRICKS_TOKEN")
    if not base_url or not token:
        sys.exit("Set DATABRICKS_BASE_URL and DATABRICKS_TOKEN (workspace serving-endpoints "
                 "URL + access token) and retry.")
    embed_model = os.environ.get("DATABRICKS_EMBED_MODEL", "databricks-bge-large-en")
    return DatabricksProvider(base_url, token, llm_model=model, embed_model=embed_model)

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Corpus-grounded thesaurus builder (monthly)")
    ap.add_argument("--csv",       default="sample_comments.csv")
    ap.add_argument("--text-col",  default="comment")
    ap.add_argument("--topic-col", default="topic")
    ap.add_argument("--out",       default="thesaurus.json")
    ap.add_argument("--gen-model",      default="databricks-meta-llama-3-3-70b-instruct",
                    help="Databricks serving endpoint for PASS 1 expansion")
    ap.add_argument("--validate-model", default="databricks-mixtral-8x7b-instruct",
                    help="Databricks serving endpoint for PASS 2 reverse-validation "
                         "(use a DIFFERENT model for independent validation)")
    ap.add_argument("--max-comments-per-topic", type=int, default=MAX_COMMENTS_PER_TOPIC)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the reverse-validation pass (expansion only)")
    ap.add_argument("--discriminative-filter", action="store_true",
                    help="drop terms that hit OTHER topics more than their own")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the result + diff but do not write the file")
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.text_col, args.topic_col])
    for col in (args.text_col, args.topic_col):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Have: {list(df.columns)}")

    topics = sorted(df[args.topic_col].unique())
    all_proc = [default_process(str(t)) or str(t).lower() for t in df[args.text_col]]
    print(f"Building thesaurus from {len(df)} comments across {len(topics)} topics")
    print(f"  expand   : {args.gen_model}")
    print(f"  validate : {'(skipped)' if args.no_validate else args.validate_model}\n")

    gen_provider = make_provider(args.gen_model)
    val_provider = (None if args.no_validate
                    else (gen_provider if args.validate_model == args.gen_model
                          else make_provider(args.validate_model)))

    thesaurus: Dict[str, List[str]] = {}
    report: Dict[str, List[Dict]] = {}

    for topic in topics:
        topic_df  = df[df[args.topic_col] == topic]
        comments  = topic_df[args.text_col].astype(str).tolist()
        if len(comments) > args.max_comments_per_topic:
            comments = (topic_df.sample(args.max_comments_per_topic, random_state=SAMPLE_SEED)
                                [args.text_col].astype(str).tolist())

        print(f"• {topic}  ({len(topic_df)} comments)")
        candidates = expand_terms(gen_provider, topic, comments)
        print(f"    expanded → {len(candidates)} candidate terms")

        if val_provider is not None:
            validated = validate_terms(val_provider, topic, candidates)
            print(f"    validated → kept {len(validated)}/{len(candidates)}")
        else:
            validated = candidates

        # always keep the topic concept itself, dedup preserving order
        seen, terms = set(), []
        for t in [topic] + validated:
            k = t.lower().strip()
            if k and k not in seen:
                seen.add(k)
                terms.append(t)

        # discriminativeness (no LLM) — report, and optionally filter
        in_proc  = [default_process(c) or c.lower() for c in topic_df[args.text_col].astype(str)]
        off_proc = [p for p, tp in zip(all_proc, df[args.topic_col]) if tp != topic]
        scores = [discriminativeness(t, in_proc, off_proc) for t in terms]
        report[topic] = scores
        if args.discriminative_filter:
            kept = [s["term"] for s in scores
                    if s["term"].lower().strip() == topic.lower().strip()
                    or s["off_rate"] <= s["in_rate"]]
            dropped = [t for t in terms if t not in kept]
            if dropped:
                print(f"    discrim-filter dropped: {dropped}")
            terms = kept

        thesaurus[topic] = terms

    # ── diff vs existing ──────────────────────────────────────────────────────
    out_path = Path(args.out)
    old = {}
    if out_path.exists():
        try:
            old = {k: v for k, v in json.loads(out_path.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
        except Exception:
            old = {}

    print("\n=== DIFF vs existing thesaurus ===")
    for topic in topics:
        new_set, old_set = set(thesaurus[topic]), set(old.get(topic, []))
        added, removed = sorted(new_set - old_set), sorted(old_set - new_set)
        if added or removed:
            print(f"  {topic}:")
            if added:   print(f"      + {added}")
            if removed: print(f"      - {removed}")
    for topic in set(old) - set(topics):
        print(f"  {topic}: (topic no longer present — left unchanged in file? NO, dropped)")

    if args.dry_run:
        print("\n[dry-run] not writing. Result preview:")
        print(json.dumps(thesaurus, indent=2, ensure_ascii=False))
        return

    # ── backup + write ────────────────────────────────────────────────────────
    if out_path.exists():
        backup = out_path.with_suffix(f".{datetime.now():%Y%m%d}.bak.json")
        backup.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"\nBacked up previous → {backup.name}")

    payload = {"_comment": f"Auto-generated by build_thesaurus.py on {datetime.now():%Y-%m-%d} "
                           f"(expand={args.gen_model}, validate="
                           f"{'none' if args.no_validate else args.validate_model}). "
                           f"True synonyms only; grounded in corpus vocabulary."}
    payload.update(thesaurus)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(out_path.stem + "_discrim_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}  ({sum(len(v) for v in thesaurus.values())} terms across "
          f"{len(thesaurus)} topics)")
    print(f"Wrote {out_path.stem}_discrim_report.json  (per-term in/off-topic hit rates)")


if __name__ == "__main__":
    main()
