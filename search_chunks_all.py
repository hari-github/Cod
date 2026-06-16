#!/usr/bin/env python
# search_chunks_all.py
# =============================================================================
# Self-contained: sentence-chunk search + threshold calibration, all in one file.
#
# Pipeline (unit = sentence chunk; results rolled up to whole comments):
#   query -> expand -> fuzzy + semantic (centered cosine) -> RRF -> rerank cascade
#         -> roll matching chunks up to unique comments
#
# Subcommands:
#   python search_chunks_all.py ingest   --csv Input.csv --text-col comment [--id-col comment_id] [--sample 200]
#   python search_chunks_all.py calibrate
#   python search_chunks_all.py search   --query "prior auth"      (omit --query for interactive)
#
# Tweak the CONSTANTS block and the CHUNKING / PROMPTS to experiment.
# Requires: pip install google-generativeai numpy pandas rapidfuzz
# =============================================================================

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

try:
    sys.stdout.reconfigure(encoding="utf-8")   # no-op in notebooks; fixes Windows console
except Exception:
    pass

# ── CONSTANTS (tune these) ────────────────────────────────────────────────────
FUZZY_THRESHOLD        = 70     # token_set_ratio (0-100) to become a candidate
FUZZY_PHRASE_QUALIFY   = 90     # contiguous-phrase score (0-100) for the LLM bypass
SEMANTIC_MIN_COSINE    = 0.20   # centered-cosine gate to enter the pool
SEMANTIC_AUTO_QUALIFY  = 0.30   # centered-cosine alone -> RELEVANT (no LLM)
FUZZY_BYPASS_MIN_COSINE = 0.20  # phrase match must also clear this centered cosine
CLASSIFY_BATCH         = 15     # comments per LLM classify call
RRF_K                  = 60     # RRF constant
MIN_CHUNK_WORDS        = 6      # merge short sentences forward to keep context
EMBED_BATCH            = 50     # texts per embedding API call

CACHE_DIR   = "./chunk_cache"
EMBED_MODEL = "gemini-embedding-2"
LLM_MODEL   = "gemma-4-31b-it"

DEFAULT_TEST_QUERIES = [
    "prior auth", "copay", "billing error", "formulary tier change", "claim denied",
    "specialty pharmacy", "telehealth", "deductible", "step therapy", "appeal process",
]

# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER  (Google AI: gemini-embedding-2 + gemma).  Swap this class to retarget.
# ═════════════════════════════════════════════════════════════════════════════
import google.generativeai as genai


class Provider:
    name = "GeminiGemma"
    embed_model = EMBED_MODEL
    llm_model = LLM_MODEL

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        res = genai.embed_content(model=self.embed_model, content=texts, task_type=task_type)
        emb = res["embedding"]
        if emb and isinstance(emb[0], (int, float)):   # single item returns a flat vector
            return [emb]
        return emb

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        return self.embed_texts([text], task_type)[0]

    def llm_call(self, system: str, user: str) -> str:
        time.sleep(2)                                   # rate-limit guard
        last = RuntimeError("no attempt")
        for attempt in range(3):
            try:
                model = genai.GenerativeModel(model_name=self.llm_model, system_instruction=system)
                return model.generate_content(user).text
            except Exception as exc:
                last = exc
                if attempt < 2:
                    print(f"  [WARN] LLM failed ({type(exc).__name__}); retry in 2s")
                    time.sleep(2)
        raise last


def get_provider(api_key: Optional[str] = None) -> Provider:
    key = api_key or os.environ.get("GEMINI_API_KEY") or _key_from_dotenv()
    if not key:
        raise SystemExit("No API key — pass --api-key, set GEMINI_API_KEY, or add it to .env")
    return Provider(key)


def _key_from_dotenv() -> str:
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.strip().split("=", 1)[1]
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# JSON / MATH HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def parse_json(raw: str, anchor: str = "") -> dict:
    """Robust to markdown fences and Gemma wrapping JSON in prose."""
    s = raw.strip()
    for fence in ("```json", "```"):
        if s.startswith(fence):
            s = s[len(fence):]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    if anchor:
        m = re.search(r'\{"' + re.escape(anchor) + r'"[\s\S]*\}', s)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("no JSON object found", s, 0)


def embedding_mean(matrix: np.ndarray) -> np.ndarray:
    """Corpus mean (from raw embeddings) — subtracted to undo anisotropy."""
    return matrix.mean(axis=0).astype(np.float32)


def center_normalize(matrix: np.ndarray, mean: np.ndarray) -> np.ndarray:
    c = matrix - mean
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)


def center_normalize_query(vec, mean: Optional[np.ndarray]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if mean is not None:
        v = v - mean
    return v / (np.linalg.norm(v) + 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION (one LLM call -> equivalents + variants)
# ═════════════════════════════════════════════════════════════════════════════
_EXPAND_SYSTEM = """\
You are a US health insurance member-experience search specialist.
Given a search query produce BOTH lists:
1. "equivalents": exact synonyms / abbreviations / spellings of the query term
   (for lexical matching). Exclude related-but-different processes.
2. "variants": 6-8 member-voice paraphrases of the SAME problem (for embedding
   search), 8-20 words each. Exclude adjacent topics.
Return ONLY this JSON, nothing else:
{"equivalents": ["..."], "variants": ["..."]}
"""


def generate_expansions(provider: Provider, query: str) -> Tuple[List[str], List[str]]:
    equivalents: List[str] = []
    variants: List[str] = []
    try:
        data = parse_json(provider.llm_call(_EXPAND_SYSTEM, f'Search query: "{query}"'),
                          anchor="equivalents")
        equivalents = [t for t in data.get("equivalents", []) if isinstance(t, str)]
        variants = [v for v in data.get("variants", []) if isinstance(v, str)]
    except Exception as e:
        print(f"  [WARN] expansion failed ({type(e).__name__}) — using bare query")
    seen, deduped = set(), []
    for t in [query] + equivalents:
        k = t.lower().strip()
        if k and k not in seen:
            seen.add(k)
            deduped.append(t)
    return deduped, (variants or [query])


# ═════════════════════════════════════════════════════════════════════════════
# FUZZY LAYER (lexical, punctuation-stripped, verbatim text)
# ═════════════════════════════════════════════════════════════════════════════
def fuzzy_search(terms: List[str], documents: List[Dict], threshold: int = FUZZY_THRESHOLD) -> List[Dict]:
    proc_terms_all = [utils.default_process(t) for t in terms]
    keep = [i for i, t in enumerate(proc_terms_all) if t]
    if not keep or not documents:
        return []
    proc_terms = [proc_terms_all[i] for i in keep]
    kept_terms = [terms[i] for i in keep]
    proc_texts = [utils.default_process(d["text"]) for d in documents]

    ts = process.cdist(proc_terms, proc_texts, scorer=fuzz.token_set_ratio, workers=-1)
    ph = np.zeros_like(ts)
    long_ids = [i for i, t in enumerate(proc_terms) if len(t) >= 5]
    short_ids = [i for i, t in enumerate(proc_terms) if len(t) < 5]
    if long_ids:
        ph[long_ids] = process.cdist([proc_terms[i] for i in long_ids], proc_texts,
                                     scorer=fuzz.partial_ratio, workers=-1)
    for i in short_ids:                              # avoid "PA" matching inside "package"
        pat = re.compile(r"\b" + re.escape(proc_terms[i]) + r"\b")
        ph[i] = [100.0 if pat.search(t) else 0.0 for t in proc_texts]

    best_ts, best_term, best_ph = ts.max(0), ts.argmax(0), ph.max(0)
    out = []
    for j, doc in enumerate(documents):
        if best_ts[j] >= threshold or best_ph[j] >= FUZZY_PHRASE_QUALIFY:
            out.append({**doc, "fuzzy_score": float(best_ts[j]) / 100.0,
                        "phrase_score": float(best_ph[j]),
                        "matched_term": kept_terms[int(best_term[j])], "layers": {"fuzzy"}})
    return sorted(out, key=lambda x: x["fuzzy_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC LAYER (dense, centered cosine, RRF across variants)
# ═════════════════════════════════════════════════════════════════════════════
def semantic_search(variants: List[str], provider: Provider, doc_matrix: np.ndarray,
                    documents: List[Dict], min_cosine: float = SEMANTIC_MIN_COSINE,
                    query_mean: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray]:
    try:
        qvecs = provider.embed_texts(variants, task_type="retrieval_query")
    except Exception as e:
        print(f"  [WARN] batch embed failed ({type(e).__name__}); per-variant fallback")
        qvecs = [provider.embed_text(v, task_type="retrieval_query") for v in variants]

    rrf: Dict[int, float] = {}
    best = np.full(len(documents), -1.0, dtype=np.float32)
    for vec in qvecs:
        q = center_normalize_query(vec, query_mean)
        sims = doc_matrix @ q
        best = np.maximum(best, sims)
        above = np.where(sims >= min_cosine)[0]
        ranked = above[np.argsort(-sims[above])]
        for rank, idx in enumerate(ranked):
            rrf[int(idx)] = rrf.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)

    hits = []
    for idx, score in sorted(rrf.items(), key=lambda x: -x[1]):
        hits.append({**documents[idx], "semantic_rrf": score,
                     "semantic_score": float(best[idx]), "layers": {"semantic"}})
    return hits, best


# ═════════════════════════════════════════════════════════════════════════════
# RRF MERGE
# ═════════════════════════════════════════════════════════════════════════════
def rrf_merge(fuzzy_hits: List[Dict], semantic_hits: List[Dict],
              best_cosines: Optional[np.ndarray], documents: List[Dict]) -> Tuple[List[Dict], Dict]:
    scores: Dict[int, float] = {}
    merged: Dict[int, Dict] = {}
    for rank, r in enumerate(fuzzy_hits):
        d = r["doc_id"]
        scores[d] = scores.get(d, 0.0) + 1.0 / (RRF_K + rank)
        merged[d] = dict(r)
    for rank, r in enumerate(semantic_hits):
        d = r["doc_id"]
        scores[d] = scores.get(d, 0.0) + 1.0 / (RRF_K + rank)
        if d in merged:
            merged[d]["layers"].add("semantic")
            merged[d]["semantic_score"] = r.get("semantic_score", 0.0)
        else:
            merged[d] = dict(r)
    for d, r in merged.items():
        r["rrf_score"] = scores[d]
    if best_cosines is not None:
        idx_of = {doc["doc_id"]: i for i, doc in enumerate(documents)}
        for d, r in merged.items():
            if "semantic_score" not in r and d in idx_of:
                r["semantic_score"] = float(best_cosines[idx_of[d]])
    counts = {"fuzzy": len(fuzzy_hits), "semantic": len(semantic_hits), "total": len(merged)}
    return sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True), counts


# ═════════════════════════════════════════════════════════════════════════════
# RERANK CASCADE (high-cosine / phrase bypass / LLM classify)
# ═════════════════════════════════════════════════════════════════════════════
_RERANK_SYSTEM = ("You are a health insurance member-experience analyst. Classify each comment "
                  "as RELEVANT or NOT_RELEVANT to the query. Default NOT_RELEVANT. Output ONLY JSON.")


def _classify_batch(batch: List[Dict], query: str, provider: Provider) -> List[Dict]:
    formatted = "\n\n".join(f"[doc_id={d['doc_id']}]\n{d['text']}" for d in batch)
    user = (f'Search query: "{query}"\n\n'
            f'For each comment, output RELEVANT if its main topic is directly about "{query}", '
            f'else NOT_RELEVANT.\n\nComments:\n\n{formatted}\n\n'
            f'Respond ONLY with this JSON, no other text:\n'
            f'{{"decisions":[{{"doc_id":1,"decision":"RELEVANT"}}]}}')
    raw = ""
    try:
        raw = provider.llm_call(_RERANK_SYSTEM, user)
        decisions = {int(d["doc_id"]): d["decision"]
                     for d in parse_json(raw, anchor="decisions").get("decisions", [])}
        return [c for c in batch if decisions.get(int(c["doc_id"]), "NOT_RELEVANT") == "RELEVANT"]
    except Exception as e:
        print(f"  [WARN] classify failed: {type(e).__name__}: {e}")
        if raw:
            print(f"  [WARN] raw[:300]: {raw[:300]}")
        print(f"  [WARN] dropping {len(batch)} candidates (NOT marking relevant)")
        return []                                    # drop, never flood


def rerank(candidates: List[Dict], query: str, provider: Provider) -> List[Dict]:
    if not candidates:
        return []
    auto, needs = [], []
    for c in candidates:
        cos = c.get("semantic_score", 0.0)
        if cos >= SEMANTIC_AUTO_QUALIFY:
            c["qualified_by"] = "high-cosine"; auto.append(c)
        elif c.get("phrase_score", 0.0) >= FUZZY_PHRASE_QUALIFY and cos >= FUZZY_BYPASS_MIN_COSINE:
            c["qualified_by"] = "phrase-match"; auto.append(c)
        else:
            needs.append(c)
    if auto:
        nh = sum(1 for c in auto if c["qualified_by"] == "high-cosine")
        print(f"        {len(auto)} auto-qualified (high-cosine {nh} | phrase {len(auto)-nh})")
    batches = [needs[i:i + CLASSIFY_BATCH] for i in range(0, len(needs), CLASSIFY_BATCH)]
    if batches:
        print(f"        {len(needs)} -> {len(batches)} LLM batch(es)")
    results = list(auto)
    for i, b in enumerate(batches, 1):
        print(f"        batch {i}/{len(batches)}")
        results.extend(_classify_batch(b, query, provider))
    return sorted(results, key=lambda x: x["rrf_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# SENTENCE CHUNKING (deterministic, verbatim)
# ═════════════════════════════════════════════════════════════════════════════
_ABBREV = r'\b(?:Dr|Mr|Mrs|Ms|vs|etc|e\.g|i\.e|U\.S|a\.m|p\.m|St|approx|No|Inc|Co|Rd|Ave)\.'


def split_sentences(text: str) -> List[str]:
    t = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
    t = re.sub(_ABBREV, lambda m: m.group(0).replace('.', '<DOT>'), t, flags=re.IGNORECASE)
    return [p.replace('<DOT>', '.').strip() for p in re.split(r'(?<=[.!?])\s+', t) if p.strip()]


def chunk_text(text: str, min_words: int = MIN_CHUNK_WORDS) -> List[str]:
    sents, chunks, buf = split_sentences(text), [], ""
    for s in sents:
        buf = f"{buf} {s}".strip() if buf else s
        if len(buf.split()) >= min_words:
            chunks.append(buf); buf = ""
    if buf:
        if chunks:
            chunks[-1] = f"{chunks[-1]} {buf}"
        else:
            chunks.append(buf)
    return chunks


# ═════════════════════════════════════════════════════════════════════════════
# CACHE I/O
# ═════════════════════════════════════════════════════════════════════════════
def _npy_path() -> Path:
    tag = EMBED_MODEL.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_")
    return Path(CACHE_DIR) / f"chunks_{tag}.npy"


def load_cache() -> Tuple[np.ndarray, np.ndarray, List[Dict], Dict[int, str]]:
    npy = _npy_path()
    if not npy.exists():
        raise SystemExit(f"No cache at {npy} — run `ingest` first.")
    raw = np.load(str(npy)).astype(np.float32)
    mean = embedding_mean(raw)
    matrix = center_normalize(raw, mean)
    chunk_docs = json.load(open(Path(CACHE_DIR) / "documents.json", encoding="utf-8"))
    comments = {c["comment_id"]: c["text"]
                for c in json.load(open(Path(CACHE_DIR) / "comments.json", encoding="utf-8"))}
    return matrix, mean, chunk_docs, comments


# ═════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═════════════════════════════════════════════════════════════════════════════
def cmd_ingest(args):
    provider = get_provider(args.api_key)
    df = pd.read_csv(args.csv).dropna(subset=[args.text_col]).reset_index(drop=True)
    if args.sample:
        df = df.head(args.sample)
    id_ok = args.id_col and args.id_col in df.columns
    comments = [{"comment_id": int(df[args.id_col].iloc[i]) if id_ok else i + 1,
                 "text": str(df[args.text_col].iloc[i])} for i in range(len(df))]

    chunks = []
    for c in comments:
        for ck in chunk_text(c["text"]):
            chunks.append({"doc_id": len(chunks) + 1, "parent_id": c["comment_id"], "text": ck})
    per = [len(chunk_text(c["text"])) for c in comments]
    print(f"{len(comments)} comments -> {len(chunks)} chunks (avg {np.mean(per):.2f}/comment)")

    texts = [c["text"] for c in chunks]
    vecs = []
    for i in range(0, len(texts), EMBED_BATCH):
        b = texts[i:i + EMBED_BATCH]
        try:
            vecs.extend(provider.embed_texts(b, task_type="retrieval_document"))
        except Exception as e:
            print(f"  [WARN] batch {i} failed ({type(e).__name__}); per-item")
            vecs.extend(provider.embed_text(t, task_type="retrieval_document") for t in b)
        print(f"  embedded {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
        time.sleep(2)
    matrix = np.array(vecs, dtype=np.float32)

    cache = Path(CACHE_DIR); cache.mkdir(parents=True, exist_ok=True)
    np.save(str(_npy_path()), matrix)
    json.dump(chunks, open(cache / "documents.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(comments, open(cache / "comments.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"doc_count": len(chunks), "comments": len(comments),
               "embed_model": EMBED_MODEL, "kind": "sentence-chunks"},
              open(_npy_path().with_suffix(".meta.json"), "w", encoding="utf-8"), indent=2)
    print(f"Saved cache -> {_npy_path().name} ({matrix.shape})")


def cmd_calibrate(args):
    provider = get_provider(args.api_key)
    raw = np.load(str(_npy_path())).astype(np.float32)
    docs = json.load(open(Path(CACHE_DIR) / "documents.json", encoding="utf-8"))
    mean = embedding_mean(raw)
    mn = center_normalize(raw, mean)

    def pair_stats(mat):
        u = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        rng = np.random.default_rng(0)
        i, j = rng.integers(0, len(u), 4000), rng.integers(0, len(u), 4000)
        k = i != j
        c = np.sum(u[i[k]] * u[j[k]], axis=1)
        return float(np.median(c)), float(np.percentile(c, 95))
    b_med, b95 = pair_stats(raw)
    a_med, a95 = pair_stats(raw - mean)
    print("ANISOTROPY (random-pair cosine):")
    print(f"  before centering: median={b_med:+.3f} p95={b95:.3f}")
    print(f"  after  centering: median={a_med:+.3f} p95={a95:.3f}")
    print(f"  -> centering {'HELPS' if a_med < 0.15 and (a95-a_med) > (b95-b_med) else 'marginal'}")

    queries = args.queries.split("||") if args.queries else DEFAULT_TEST_QUERIES
    qvecs = provider.embed_texts(queries, task_type="retrieval_query")
    floors, autos = [], []
    print("\nPER-QUERY (top docs + noise model):")
    for q, qv in zip(queries, qvecs):
        cos = mn @ center_normalize_query(qv, mean)
        med = float(np.median(cos))
        mad = float(np.median(np.abs(cos - med))) * 1.4826 + 1e-9
        floors.append(med + 3 * mad); autos.append(med + 5 * mad)
        order = np.argsort(-cos)
        print(f"\n  [{q}] max={cos.max():.3f} p95={np.percentile(cos,95):.3f} "
              f"noise={med:+.3f} mad={mad:.3f}")
        for idx in order[:args.top_n]:
            print(f"     {cos[idx]:+.3f} {docs[idx]['text'][:66]}")
    print("\nSUGGESTED (verify against top-N text above):")
    print(f"  SEMANTIC_MIN_COSINE     ~ {round(float(np.median(floors)),2)}  (current {SEMANTIC_MIN_COSINE})")
    print(f"  FUZZY_BYPASS_MIN_COSINE ~ {round(float(np.median(floors)),2)}  (current {FUZZY_BYPASS_MIN_COSINE})")
    print(f"  SEMANTIC_AUTO_QUALIFY   ~ {round(float(np.median(autos)),2)}  (current {SEMANTIC_AUTO_QUALIFY})")


def run_query(provider, matrix, mean, chunk_docs, comments, query, verbose=True):
    equivalents, variants = generate_expansions(provider, query)
    fz = fuzzy_search(equivalents, chunk_docs)
    sm, best = semantic_search(variants, provider, matrix, chunk_docs, query_mean=mean)
    merged, counts = rrf_merge(fz, sm, best, chunk_docs)
    if verbose:
        print(f"        candidates: fuzzy {counts['fuzzy']} | semantic {counts['semantic']} "
              f"| total {counts['total']}")
    relevant = rerank(merged, query, provider)
    by_parent: Dict[int, Dict] = {}
    for ch in relevant:
        p = ch["parent_id"]
        if p not in by_parent or ch["rrf_score"] > by_parent[p]["rrf_score"]:
            by_parent[p] = ch
    return sorted(by_parent.values(), key=lambda x: -x["rrf_score"]), len(relevant)


def cmd_search(args):
    provider = get_provider(args.api_key)
    matrix, mean, chunk_docs, comments = load_cache()
    print(f"Loaded {len(chunk_docs)} chunks from {len(comments)} comments.")

    def do(q):
        results, n_chunks = run_query(provider, matrix, mean, chunk_docs, comments, q)
        print(f"\n'{q}' -> {n_chunks} relevant chunks -> {len(results)} comments\n")
        for i, ch in enumerate(results, 1):
            print(f"#{i} comment {ch['parent_id']} (cosine {ch.get('semantic_score',0):.2f}, "
                  f"{ch.get('qualified_by','llm')})")
            print(f"   matched: {ch['text'][:100]}")
            print(f"   comment: {comments[ch['parent_id']][:140]}\n")

    if args.query:
        do(args.query)
    else:
        print("Interactive search — blank line to quit.")
        while True:
            try:
                q = input("\nsearch > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            do(q)


def main():
    ap = argparse.ArgumentParser(description="Sentence-chunk search + calibration (self-contained)")
    ap.add_argument("--api-key", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.set_defaults(func=cmd_ingest)
    p.add_argument("--csv", required=True); p.add_argument("--text-col", default="comment")
    p.add_argument("--id-col", default=None); p.add_argument("--sample", type=int, default=None)

    p = sub.add_parser("calibrate"); p.set_defaults(func=cmd_calibrate)
    p.add_argument("--queries", default=None, help='queries joined by "||"; default uses built-in set')
    p.add_argument("--top-n", type=int, default=12)

    p = sub.add_parser("search"); p.set_defaults(func=cmd_search)
    p.add_argument("--query", default=None)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
