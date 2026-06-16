#!/usr/bin/env python
# search_summaries_all.py
# =============================================================================
# Self-contained: per-topic SUMMARY search, all in one file.
#
# Idea: at ingest, an LLM splits each comment into one self-contained summary
# PER TOPIC. Search runs:
#     - SEMANTIC + LLM CLASSIFY on the summaries (clean, single-topic units)
#     - FUZZY on the ORIGINAL comment text (verbatim, so exact terms still match)
# Matching summaries are rolled up to unique parent comments (whole comment returned).
#
# Subcommands:
#   python search_summaries_all.py ingest    --csv Input.csv --text-col comment [--id-col comment_id] [--sample 100]
#   python search_summaries_all.py calibrate
#   python search_summaries_all.py search     --query "prior auth"   (omit for interactive)
#
# NOTE: ingest makes ONE LLM call per comment (slow + rate-limited). Use --sample first.
# Requires: pip install google-generativeai numpy pandas rapidfuzz
# =============================================================================

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process, utils

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── CONSTANTS (tune these) ────────────────────────────────────────────────────
FUZZY_THRESHOLD        = 70
FUZZY_PHRASE_QUALIFY   = 90
SEMANTIC_MIN_COSINE    = 0.20
SEMANTIC_AUTO_QUALIFY  = 0.30
FUZZY_BYPASS_MIN_COSINE = 0.20
CLASSIFY_BATCH         = 15
RRF_K                  = 60
EMBED_BATCH            = 50
SUMMARY_BATCH          = 5      # comments per summary LLM call (mapped back by id)

CACHE_DIR   = "./summary_cache"
EMBED_MODEL = "gemini-embedding-2"
LLM_MODEL   = "gemma-4-31b-it"

DEFAULT_TEST_QUERIES = [
    "prior auth", "copay", "billing error", "formulary tier change", "claim denied",
    "specialty pharmacy", "telehealth", "deductible", "step therapy", "appeal process",
]

# ═════════════════════════════════════════════════════════════════════════════
# PROVIDER (Google AI). Swap this class to retarget.
# ═════════════════════════════════════════════════════════════════════════════
import google.generativeai as genai


class GeminiProvider:
    """Google AI: gemini-embedding-2 + gemma (default)."""
    name = "GeminiGemma"

    def __init__(self, api_key: str, embed_model: str = EMBED_MODEL, llm_model: str = LLM_MODEL):
        genai.configure(api_key=api_key)
        self.embed_model = embed_model
        self.llm_model = llm_model

    def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        res = genai.embed_content(model=self.embed_model, content=texts, task_type=task_type)
        emb = res["embedding"]
        if emb and isinstance(emb[0], (int, float)):
            return [emb]
        return emb

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        return self.embed_texts([text], task_type)[0]

    def llm_call(self, system: str, user: str) -> str:
        time.sleep(2)
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


class DatabricksProvider:
    """OpenAI-compatible Databricks Model Serving endpoint (embeddings + chat)."""
    name = "Databricks"

    def __init__(self, base_url: str, token: str, llm_model: str, embed_model: str):
        from openai import OpenAI                       # imported lazily so Gemini users need no openai
        self._client = OpenAI(api_key=token, base_url=base_url)
        self.llm_model = llm_model
        self.embed_model = embed_model

    def embed_texts(self, texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
        resp = self._client.embeddings.create(model=self.embed_model, input=list(texts))
        return [list(d.embedding) for d in resp.data]   # OpenAI API preserves input order

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        return self.embed_texts([text], task_type)[0]

    def llm_call(self, system: str, user: str) -> str:
        last = RuntimeError("no attempt")
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.llm_model, temperature=0.0,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return resp.choices[0].message.content
            except Exception as exc:
                last = exc
                if attempt < 2:
                    print(f"  [WARN] LLM failed ({type(exc).__name__}); retry in 2s")
                    time.sleep(2)
        raise last


Provider = Union[GeminiProvider, DatabricksProvider]


def _key_from_dotenv() -> str:
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.strip().split("=", 1)[1]
    return ""


def get_provider(args) -> Provider:
    """Build the provider chosen by --provider (gemini | databricks)."""
    if getattr(args, "provider", "gemini") == "databricks":
        base  = args.db_base_url   or os.environ.get("DATABRICKS_BASE_URL")
        token = args.db_token      or os.environ.get("DATABRICKS_TOKEN")
        llm   = args.db_llm_model  or os.environ.get("DATABRICKS_LLM_MODEL")
        emb   = args.db_embed_model or os.environ.get("DATABRICKS_EMBED_MODEL")
        missing = [n for n, v in [("--db-base-url", base), ("--db-token", token),
                                  ("--db-llm-model", llm), ("--db-embed-model", emb)] if not v]
        if missing:
            raise SystemExit("Databricks needs: " + ", ".join(missing) +
                             "  (pass as args or set DATABRICKS_BASE_URL/TOKEN/LLM_MODEL/EMBED_MODEL).")
        return DatabricksProvider(base, token, llm, emb)
    key = args.api_key or os.environ.get("GEMINI_API_KEY") or _key_from_dotenv()
    if not key:
        raise SystemExit("Gemini needs --api-key, GEMINI_API_KEY, or .env")
    return GeminiProvider(key)


# ═════════════════════════════════════════════════════════════════════════════
# JSON / MATH HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def parse_json(raw: str, anchor: str = "") -> dict:
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
# QUERY EXPANSION
# ═════════════════════════════════════════════════════════════════════════════
_EXPAND_SYSTEM = """\
You are a US health insurance member-experience search specialist.
Given a search query produce BOTH lists:
1. "equivalents": exact synonyms / abbreviations / spellings of the query term.
2. "variants": 6-8 member-voice paraphrases of the SAME problem, 8-20 words each.
Return ONLY this JSON: {"equivalents": ["..."], "variants": ["..."]}
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
# TOPIC SUMMARIES (LLM split — used only at ingest)
# ═════════════════════════════════════════════════════════════════════════════
_SUMMARY_SYSTEM = """\
You are a health insurance member-experience analyst.
For EACH member comment, split it into its DISTINCT topics and write ONE
self-contained summary (1-2 sentences) per topic that fully captures what the
member said about that topic, with enough context to stand alone. One topic ->
one summary; three topics -> three summaries. Do NOT invent topics. Keep each
comment's summaries under its own comment_id. Output ONLY JSON.
"""


def generate_summaries_batch(provider: Provider, batch: List[Dict]) -> Dict[int, List[str]]:
    """
    One LLM call for several comments. Each comment is tagged with its id; the
    response maps every comment_id to its list of topic summaries. Any comment
    missing from the response falls back to its whole text (never lose a comment).
    """
    formatted = "\n\n".join(f"[comment_id={c['comment_id']}]\n{c['text']}" for c in batch)
    user = (
        "For each comment below, return its topic summaries.\n"
        "Respond ONLY with this JSON (no other text), one entry per comment_id:\n"
        '{"results":[{"comment_id":1,"summaries":["...","..."]},'
        '{"comment_id":2,"summaries":["..."]}]}\n\n'
        f"Comments:\n\n{formatted}"
    )
    out: Dict[int, List[str]] = {}
    try:
        data = parse_json(provider.llm_call(_SUMMARY_SYSTEM, user), anchor="results")
        for r in data.get("results", []):
            try:
                cid = int(r["comment_id"])
            except (KeyError, ValueError, TypeError):
                continue
            sums = [x.strip() for x in r.get("summaries", []) if isinstance(x, str) and x.strip()]
            if sums:
                out[cid] = sums
    except Exception as e:
        print(f"  [WARN] summary batch failed ({type(e).__name__}); using whole comments")
    for c in batch:                                  # fallback for any comment the LLM dropped
        out.setdefault(c["comment_id"], [c["text"]])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# FUZZY LAYER (runs on ORIGINAL comments)
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
    for i in short_ids:
        pat = re.compile(r"\b" + re.escape(proc_terms[i]) + r"\b")
        ph[i] = [100.0 if pat.search(t) else 0.0 for t in proc_texts]

    best_ts, best_term, best_ph = ts.max(0), ts.argmax(0), ph.max(0)
    out = []
    for j, doc in enumerate(documents):
        if best_ts[j] >= threshold or best_ph[j] >= FUZZY_PHRASE_QUALIFY:
            out.append({**doc, "fuzzy_score": float(best_ts[j]) / 100.0,
                        "phrase_score": float(best_ph[j]),
                        "matched_term": kept_terms[int(best_term[j])]})
    return sorted(out, key=lambda x: x["fuzzy_score"], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC LAYER (runs on SUMMARIES)
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
        hits.append({**documents[idx], "semantic_rrf": score, "semantic_score": float(best[idx])})
    return hits, best


# ═════════════════════════════════════════════════════════════════════════════
# RERANK CASCADE (runs on SUMMARIES)
# ═════════════════════════════════════════════════════════════════════════════
_RERANK_SYSTEM = ("You are a health insurance member-experience analyst. Classify each text "
                  "as RELEVANT or NOT_RELEVANT to the query. Default NOT_RELEVANT. Output ONLY JSON.")


def _classify_batch(batch: List[Dict], query: str, provider: Provider) -> List[Dict]:
    formatted = "\n\n".join(f"[doc_id={d['doc_id']}]\n{d['text']}" for d in batch)
    user = (f'Search query: "{query}"\n\n'
            f'For each text, output RELEVANT if its main topic is directly about "{query}", '
            f'else NOT_RELEVANT.\n\nTexts:\n\n{formatted}\n\n'
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
        return []


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
# CACHE I/O
# ═════════════════════════════════════════════════════════════════════════════
def _npy_path(embed_model: str) -> Path:
    tag = embed_model.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_")
    return Path(CACHE_DIR) / f"summaries_{tag}.npy"


def load_cache(embed_model: str) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict]]:
    npy = _npy_path(embed_model)
    if not npy.exists():
        raise SystemExit(f"No cache at {npy} — run `ingest` with this provider first.")
    raw = np.load(str(npy)).astype(np.float32)
    mean = embedding_mean(raw)
    matrix = center_normalize(raw, mean)
    summaries = json.load(open(Path(CACHE_DIR) / "documents.json", encoding="utf-8"))
    comments = json.load(open(Path(CACHE_DIR) / "comments.json", encoding="utf-8"))
    return matrix, mean, summaries, comments


# ═════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ═════════════════════════════════════════════════════════════════════════════
def cmd_ingest(args):
    provider = get_provider(args)
    df = pd.read_csv(args.csv).dropna(subset=[args.text_col]).reset_index(drop=True)
    if args.sample:
        df = df.head(args.sample)
    id_ok = args.id_col and args.id_col in df.columns
    comments = [{"comment_id": int(df[args.id_col].iloc[i]) if id_ok else i + 1,
                 "text": str(df[args.text_col].iloc[i])} for i in range(len(df))]

    batch_size = args.summary_batch or SUMMARY_BATCH
    n_calls = (len(comments) + batch_size - 1) // batch_size
    print(f"Summarizing {len(comments)} comments, {batch_size}/call -> {n_calls} LLM call(s) ...")
    summaries: List[Dict] = []
    for b in range(0, len(comments), batch_size):
        batch = comments[b:b + batch_size]
        mapping = generate_summaries_batch(provider, batch)
        for c in batch:                              # preserve input order; map by id
            for s in mapping.get(c["comment_id"], [c["text"]]):
                summaries.append({"doc_id": len(summaries) + 1,
                                  "parent_id": c["comment_id"], "text": s})
        done = min(b + batch_size, len(comments))
        print(f"  summarized {done}/{len(comments)}  (summaries so far: {len(summaries)})")
    print(f"{len(comments)} comments -> {len(summaries)} summaries "
          f"(avg {len(summaries)/max(len(comments),1):.2f}/comment)")

    texts = [s["text"] for s in summaries]
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
    npy = _npy_path(provider.embed_model)
    np.save(str(npy), matrix)
    json.dump(summaries, open(cache / "documents.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump(comments, open(cache / "comments.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    json.dump({"doc_count": len(summaries), "comments": len(comments),
               "provider": provider.name, "embed_model": provider.embed_model, "kind": "topic-summaries"},
              open(npy.with_suffix(".meta.json"), "w", encoding="utf-8"), indent=2)
    print(f"Saved cache -> {npy.name} ({matrix.shape})")


def cmd_calibrate(args):
    provider = get_provider(args)
    npy = _npy_path(provider.embed_model)
    if not npy.exists():
        raise SystemExit(f"No cache at {npy} — run `ingest` with this provider first.")
    raw = np.load(str(npy)).astype(np.float32)
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
    print("ANISOTROPY (random-pair cosine on summaries):")
    print(f"  before centering: median={b_med:+.3f} p95={b95:.3f}")
    print(f"  after  centering: median={a_med:+.3f} p95={a95:.3f}")
    print(f"  -> centering {'HELPS' if a_med < 0.15 and (a95-a_med) > (b95-b_med) else 'marginal'}")

    queries = args.queries.split("||") if args.queries else DEFAULT_TEST_QUERIES
    qvecs = provider.embed_texts(queries, task_type="retrieval_query")
    floors, autos = [], []
    print("\nPER-QUERY (top summaries + noise model):")
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


def run_query(provider, sum_matrix, mean, summaries, comments_by_id, query, verbose=True):
    equivalents, variants = generate_expansions(provider, query)

    comment_docs = [{"doc_id": cid, "text": txt} for cid, txt in comments_by_id.items()]
    summaries_by_parent: Dict[int, List[Dict]] = {}
    for s in summaries:
        summaries_by_parent.setdefault(s["parent_id"], []).append(s)
    idx_of = {s["doc_id"]: i for i, s in enumerate(summaries)}

    # FUZZY on original comments
    fz = fuzzy_search(equivalents, comment_docs)
    fuzzy_rank = {r["doc_id"]: rank for rank, r in enumerate(fz)}
    fuzzy_phrase = {r["doc_id"]: r["phrase_score"] for r in fz}

    # SEMANTIC on summaries
    sm, best = semantic_search(variants, provider, sum_matrix, summaries, query_mean=mean)
    sem_rank = {r["doc_id"]: rank for rank, r in enumerate(sm)}

    # Build summary candidates (semantic hits + summaries of fuzzy-matched comments)
    cand: Dict[int, Dict] = {}
    for rank, r in enumerate(sm):
        cand[r["doc_id"]] = {**r, "rrf_score": 1.0 / (RRF_K + rank), "phrase_score": 0.0}
    for cid in fuzzy_rank:
        frank = fuzzy_rank[cid]
        for s in summaries_by_parent.get(cid, []):
            sid = s["doc_id"]
            if sid in cand:
                cand[sid]["rrf_score"] += 1.0 / (RRF_K + frank)
                cand[sid]["phrase_score"] = max(cand[sid]["phrase_score"], fuzzy_phrase[cid])
            else:
                cand[sid] = {**s, "rrf_score": 1.0 / (RRF_K + frank),
                             "semantic_score": float(best[idx_of[sid]]),
                             "phrase_score": fuzzy_phrase[cid]}
    candidates = sorted(cand.values(), key=lambda x: -x["rrf_score"])
    if verbose:
        print(f"        candidates: fuzzy-comments {len(fz)} | semantic-summaries {len(sm)} "
              f"| total summaries {len(candidates)}")

    # RERANK + CLASSIFY on summaries, then roll up to comments
    relevant = rerank(candidates, query, provider)
    by_parent: Dict[int, Dict] = {}
    for s in relevant:
        p = s["parent_id"]
        if p not in by_parent or s["rrf_score"] > by_parent[p]["rrf_score"]:
            by_parent[p] = s
    return sorted(by_parent.values(), key=lambda x: -x["rrf_score"]), len(relevant)


def cmd_search(args):
    provider = get_provider(args)
    matrix, mean, summaries, comments = load_cache(provider.embed_model)
    comments_by_id = {c["comment_id"]: c["text"] for c in comments}
    print(f"Loaded {len(summaries)} summaries from {len(comments)} comments ({provider.name}).")

    def do(q):
        results, n_sum = run_query(provider, matrix, mean, summaries, comments_by_id, q)
        print(f"\n'{q}' -> {n_sum} relevant summaries -> {len(results)} comments\n")
        for i, s in enumerate(results, 1):
            print(f"#{i} comment {s['parent_id']} (cosine {s.get('semantic_score',0):.2f}, "
                  f"{s.get('qualified_by','llm')})")
            print(f"   matched summary: {s['text'][:110]}")
            print(f"   full comment   : {comments_by_id[s['parent_id']][:140]}\n")

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
    ap = argparse.ArgumentParser(description="Topic-summary search (self-contained)")
    # global provider options (place BEFORE the subcommand on the command line)
    ap.add_argument("--provider", choices=["gemini", "databricks"], default="gemini")
    ap.add_argument("--api-key", default=None, help="Gemini API key (or GEMINI_API_KEY / .env)")
    ap.add_argument("--db-base-url", default=None, help="Databricks base URL (or DATABRICKS_BASE_URL)")
    ap.add_argument("--db-token", default=None, help="Databricks token (or DATABRICKS_TOKEN)")
    ap.add_argument("--db-llm-model", default=None, help="Databricks LLM endpoint (or DATABRICKS_LLM_MODEL)")
    ap.add_argument("--db-embed-model", default=None, help="Databricks embed endpoint (or DATABRICKS_EMBED_MODEL)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ingest"); p.set_defaults(func=cmd_ingest)
    p.add_argument("--csv", required=True); p.add_argument("--text-col", default="comment")
    p.add_argument("--id-col", default=None); p.add_argument("--sample", type=int, default=None)
    p.add_argument("--summary-batch", type=int, default=None,
                   help=f"comments per summary LLM call (default {SUMMARY_BATCH})")

    p = sub.add_parser("calibrate"); p.set_defaults(func=cmd_calibrate)
    p.add_argument("--queries", default=None, help='queries joined by "||"; default uses built-in set')
    p.add_argument("--top-n", type=int, default=12)

    p = sub.add_parser("search"); p.set_defaults(func=cmd_search)
    p.add_argument("--query", default=None)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
