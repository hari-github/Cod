# calibrate_thresholds.py — Validate & calibrate search_v2 cosine thresholds
#
# Run cell by cell in VS Code (Shift+Enter on each # %% block).
#
# Why this exists:
#   search_v2's cosine thresholds (SEMANTIC_MIN_COSINE, SEMANTIC_AUTO_QUALIFY,
#   FUZZY_BYPASS_MIN_COSINE) live on the MEAN-CENTERED scale and were calibrated
#   on a 200-doc corpus. Different corpus size / content shifts the score
#   distribution, so re-validate before trusting them. This tool:
#     1. Confirms centering still helps on the new data (anisotropy check)
#     2. Shows where relevant docs sit vs noise, per query (eyeball calibration)
#     3. Suggests thresholds from the noise statistics (corpus-size robust)
#     4. (optional) Computes precision/recall/F1 if you supply labels
#
#   Note: FUZZY_THRESHOLD (70) and FUZZY_PHRASE_QUALIFY (90) are on the 0-100
#   rapidfuzz scale — string matching, corpus-independent — so they do NOT need
#   recalibration. Only the cosine thresholds depend on the embedding scale.
#
# Requires: pip install google-generativeai numpy   (+ the existing project files)

# %% ── CELL 1: Configuration ──────────────────────────────────────────────────
# Point this at the cache you built for the 2000-comment data, then edit the
# test queries to match topics that actually appear in your comments.

CACHE_DIR      = "./embed_cache"   # folder holding *.npy + *.meta.json
GEMINI_API_KEY = ""                # leave "" to read from .env (used if you pick
                                   # the Gemini / OllamaGemini provider in CELL 2)
TOP_N          = 15                # how many top docs to print per query
# Provider is chosen interactively in CELL 2 — same 1/2/3/4 menu as search_v2.py.

# Representative queries spanning your data's topics. More queries = a more
# stable suggestion. Use real phrasings members/analysts would search.
TEST_QUERIES = [
    "prior auth",
    "copay",
    "billing error",
    "formulary tier change",
    "claim denied",
    "specialty pharmacy",
    "telehealth",
    "deductible",
    "step therapy",
    "appeal process",
]

# OPTIONAL ground truth for rigorous calibration (CELL 7). Map each query to the
# set of doc_ids that are genuinely relevant. Leave {} to skip labeled eval and
# rely on the eyeball + statistical suggestion instead.
#   LABELS = {"prior auth": [1, 2, 5, 7, 11], "copay": [40, 41]}
LABELS: dict = {}

print("Config set.")
print(f"  Cache dir : {CACHE_DIR}")
print(f"  Queries   : {len(TEST_QUERIES)}")
print(f"  Labels    : {len(LABELS)} querie(s) labeled" if LABELS else "  Labels    : none (eyeball + stats mode)")

# %% ── CELL 2: Imports & load the centered index ───────────────────────────────

import json
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252 and choke on arrows/sigma in the output.
# Force UTF-8 when possible (no-op in the VS Code interactive window).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os

from search_v2 import (
    SEMANTIC_MIN_COSINE, SEMANTIC_AUTO_QUALIFY, FUZZY_BYPASS_MIN_COSINE,
    embedding_mean, center_normalize, center_normalize_query, select_provider,
)


def _resolve_api_key() -> str:
    """Best-effort Gemini key from CELL 1 or .env — returns '' if unset."""
    if GEMINI_API_KEY:
        return GEMINI_API_KEY
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.strip().split("=", 1)[1]
    return ""


def _find_cache(cache_dir: str) -> Path:
    d = Path(cache_dir)
    npys = sorted(
        [f for f in d.glob("*.npy") if f.with_suffix(".meta.json").exists()],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    if not npys:
        raise SystemExit(f"No *.npy cache with a .meta.json found in {cache_dir}")
    return npys[0]


npy_path  = _find_cache(CACHE_DIR)
meta      = json.load(open(npy_path.with_suffix(".meta.json"), encoding="utf-8"))
documents = json.load(open(Path(CACHE_DIR) / "documents.json", encoding="utf-8"))

raw_matrix = np.load(str(npy_path)).astype(np.float32)
query_mean = embedding_mean(raw_matrix)
doc_matrix = center_normalize(raw_matrix, query_mean)   # centered + normalized

# Expose the key so select_provider() won't re-prompt for Gemini / OllamaGemini.
_key = _resolve_api_key()
if _key:
    os.environ.setdefault("GEMINI_API_KEY", _key)

# Same 1/2/3/4 menu as search_v2.py (Databricks / Gemini / Local / OllamaGemini).
provider = select_provider()

# CRITICAL: query vectors must come from the SAME embedding model that built the
# cache, or they live in a different space and every cosine is meaningless.
cache_model = meta.get("embed_model", "")
if cache_model and provider.embed_model != cache_model:
    raise SystemExit(
        f"\n  Embed model mismatch — calibration would be invalid.\n"
        f"    cache was built with : {cache_model}\n"
        f"    selected provider use: {provider.embed_model}\n"
        f"  Choose the provider whose embed model matches the cache, or rebuild it."
    )

print(f"\nLoaded cache : {npy_path.name}")
print(f"  Docs       : {len(documents)}   (meta says {meta.get('doc_count', '?')})")
print(f"  Embed model: {meta.get('embed_model', '?')}   dim={raw_matrix.shape[1]}")
print(f"  Provider   : {provider.name}  (embed {provider.embed_model})")
print(f"  Current thresholds -> min={SEMANTIC_MIN_COSINE}  auto={SEMANTIC_AUTO_QUALIFY}  fuzzy_bypass={FUZZY_BYPASS_MIN_COSINE}")

# %% ── CELL 3: Anisotropy check — is centering still needed/working? ───────────
# Median cosine between two RANDOM docs should be ~0 after centering. If "before"
# is high (e.g. 0.6-0.9) and "after" drops near 0, centering is doing its job.
# Sample pairs for speed on large corpora.

def _pair_stats(mat: np.ndarray, sample: int = 4000) -> dict:
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    n  = len(mn)
    rng = np.random.default_rng(0)
    i = rng.integers(0, n, sample)
    j = rng.integers(0, n, sample)
    keep = i != j
    cos = np.sum(mn[i[keep]] * mn[j[keep]], axis=1)
    return {"median": float(np.median(cos)),
            "p95": float(np.percentile(cos, 95)),
            "spread": float(np.percentile(cos, 95) - np.median(cos))}

before = _pair_stats(raw_matrix)
after  = _pair_stats(raw_matrix - query_mean)

print("Random-pair cosine (noise floor — lower median is better):")
print(f"  BEFORE centering : median={before['median']:+.3f}  p95={before['p95']:.3f}  spread={before['spread']:.3f}")
print(f"  AFTER  centering : median={after['median']:+.3f}  p95={after['p95']:.3f}  spread={after['spread']:.3f}")
print()
if after["median"] < 0.15 and after["spread"] > before["spread"]:
    print("  [OK] Centering works on this data — random pairs recentre near 0, spread widens.")
elif before["median"] < 0.2:
    print("  [!] This model is NOT very anisotropic — centering may be optional here.")
else:
    print("  [!] Centering didn't help as expected — inspect the data before trusting thresholds.")

# %% ── CELL 4: Embed all test queries (centered) ───────────────────────────────
# One batched API call. Each query vector gets the SAME centering as the docs.

raw_qvecs = provider.embed_texts(TEST_QUERIES, task_type="retrieval_query")
qvecs = np.vstack([center_normalize_query(v, query_mean) for v in raw_qvecs])
cos_by_query = {q: doc_matrix @ qvecs[i] for i, q in enumerate(TEST_QUERIES)}
print(f"Embedded {len(TEST_QUERIES)} queries → cosine vectors of length {len(documents)} each.")

# %% ── CELL 5: Per-query distribution + top-N (eyeball calibration) ────────────
# Read the top docs: where the text stops being on-topic is your real boundary.
# The counts show how many docs each candidate threshold would admit/auto-qualify.

id_to_text = {d["doc_id"]: d["text"] for d in documents}

for q in TEST_QUERIES:
    cos = cos_by_query[q]
    order = np.argsort(-cos)
    print(f"\n{'='*78}\n  QUERY: {q}")
    print(f"  max={cos.max():.3f}  p99={np.percentile(cos,99):.3f}  "
          f"p95={np.percentile(cos,95):.3f}  median={np.percentile(cos,50):+.3f}")
    counts = "  ".join(f">={t}:{int((cos>=t).sum())}" for t in (0.20, 0.25, 0.30, 0.35, 0.40))
    print(f"  docs admitted at  {counts}")
    print(f"  top {TOP_N} (look for where it stops being about '{q}'):")
    for idx in order[:TOP_N]:
        print(f"    {cos[idx]:+.3f}  [{documents[idx]['doc_id']}] {documents[idx]['text'][:72]}")

# %% ── CELL 6: Suggested thresholds (statistical, corpus-size robust) ──────────
# Models the per-query NOISE with a robust median + MAD (median absolute
# deviation). Relevant docs are the high outliers. Suggestions:
#   floor (enter pool)  = noise_median + 3 * MAD   ("3 sigma above noise")
#   auto-qualify        = noise_median + 5 * MAD   ("clearly above noise")
# Median across queries. MAD scales with the actual noise spread, so this
# adapts automatically as corpus size / model changes -- unlike fixed numbers.

def _robust(cos: np.ndarray):
    med = np.median(cos)
    mad = np.median(np.abs(cos - med)) * 1.4826 + 1e-9   # ~std for normal noise
    return med, mad

floors, autos = [], []
for q in TEST_QUERIES:
    med, mad = _robust(cos_by_query[q])
    floors.append(med + 3 * mad)
    autos.append(med + 5 * mad)

sugg_floor = float(np.median(floors))
sugg_auto  = float(np.median(autos))

print("Per-query noise model (median, MAD) and derived cut points:")
for q in TEST_QUERIES:
    med, mad = _robust(cos_by_query[q])
    print(f"  {q:24s} noise={med:+.3f}  mad={mad:.3f}  -> 3sig={med+3*mad:.3f}  5sig={med+5*mad:.3f}")

print(f"\nSUGGESTED (median across queries, rounded to 0.01):")
print(f"  SEMANTIC_MIN_COSINE      ~ {round(sugg_floor, 2)}   (current {SEMANTIC_MIN_COSINE})")
print(f"  FUZZY_BYPASS_MIN_COSINE  ~ {round(sugg_floor, 2)}   (current {FUZZY_BYPASS_MIN_COSINE})")
print(f"  SEMANTIC_AUTO_QUALIFY    ~ {round(sugg_auto, 2)}   (current {SEMANTIC_AUTO_QUALIFY})")
print("\n  Treat these as starting points -- confirm against the CELL 5 top-N text,")
print("  then edit the constants at the top of search_v2.py.")

# %% ── CELL 7: (OPTIONAL) Labeled precision / recall / F1 sweep ────────────────
# Only runs if you filled in LABELS in CELL 1. For each cosine threshold it
# reports micro precision/recall/F1 over all labeled queries. Use it to pick:
#   SEMANTIC_MIN_COSINE   -> threshold near best F1 (balance recall vs noise)
#   SEMANTIC_AUTO_QUALIFY -> lowest threshold where precision is ~1.0
#                           (auto-qualify must be high-precision -- no LLM check)

if not LABELS:
    print("No LABELS provided — skipping labeled eval. (Fill LABELS in CELL 1 to enable.)")
else:
    sweep = [round(x, 2) for x in np.arange(0.10, 0.51, 0.025)]
    print(f"{'thr':>6} {'prec':>7} {'recall':>7} {'F1':>7}   (micro over labeled queries)")
    best = (None, -1.0)
    for t in sweep:
        tp = fp = fn = 0
        for q, rel in LABELS.items():
            if q not in cos_by_query:
                continue
            rel_set = set(rel)
            pred = {documents[i]["doc_id"] for i in np.where(cos_by_query[q] >= t)[0]}
            tp += len(pred & rel_set)
            fp += len(pred - rel_set)
            fn += len(rel_set - pred)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        flag = ""
        if f1 > best[1]:
            best = (t, f1); flag = "  <- best F1"
        if prec >= 0.999 and tp > 0:
            flag += "  (precision~1 -> safe auto-qualify)"
        print(f"{t:>6.3f} {prec:>7.2f} {rec:>7.2f} {f1:>7.2f}{flag}")
    print(f"\n  Best F1 at threshold {best[0]} -> candidate for SEMANTIC_MIN_COSINE.")
    print("  Lowest threshold with precision~1.0 -> candidate for SEMANTIC_AUTO_QUALIFY.")
