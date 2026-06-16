# ingest_chunks.py — EXPERIMENTAL: sentence-chunk ingestion + search test
#
# Splits each comment into sentence chunks (deterministic, NO LLM, verbatim text),
# embeds each chunk, then lets you search over chunks and roll results back up to
# whole comments. Compare against the whole-comment pipeline to decide if topic
# splitting helps before trying the heavier LLM-summary approach.
#
# Isolated by design:
#   - writes to ./chunk_cache (NOT ./embed_cache) so app_search.py never loads it
#   - does not import or modify the app
#
# Run cell by cell in VS Code (Shift+Enter).
# Requires: pip install google-generativeai numpy pandas rapidfuzz

# %% ── CELL 1: Configuration ──────────────────────────────────────────────────

CSV_PATH    = "Input.csv"        # same source CSV as your whole-comment cache
TEXT_COL    = "comment"
ID_COL      = "comment_id"       # set None to use row numbers

CHUNK_CACHE_DIR = "./chunk_cache"  # kept separate from ./embed_cache on purpose

SAMPLE_SIZE     = 200            # test on a sample first; set None for ALL comments
MIN_CHUNK_WORDS = 6             # merge tiny sentences forward so chunks keep context
EMBED_BATCH     = 50            # chunks per embedding API call

GEMINI_API_KEY  = ""            # leave "" to read from .env

print("Config set.")
print(f"  CSV         : {CSV_PATH}  (text='{TEXT_COL}', id='{ID_COL}')")
print(f"  Sample size : {SAMPLE_SIZE if SAMPLE_SIZE else 'ALL'}")
print(f"  Min words   : {MIN_CHUNK_WORDS} per chunk")
print(f"  Cache dir   : {CHUNK_CACHE_DIR}")

# %% ── CELL 2: Imports & provider ──────────────────────────────────────────────

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")   # no-op in VS Code interactive
except Exception:
    pass

from search_v2 import (
    FUZZY_THRESHOLD, SEMANTIC_MIN_COSINE,
    embedding_mean, center_normalize, center_normalize_query,
    fuzzy_search, generate_expansions, llm_rerank, rrf_merge, semantic_search,
)
from provider_ollama_gemini import OllamaGeminiProvider


def _resolve_api_key() -> str:
    if GEMINI_API_KEY:
        return GEMINI_API_KEY
    env = Path(".env")
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise SystemExit("No API key — set GEMINI_API_KEY in CELL 1 or in .env")


# Same embedding model (gemini-embedding-2) as the main cache, so the chunk test
# is comparable to the whole-comment results. Swap providers here if you wish.
provider = OllamaGeminiProvider(_resolve_api_key())
print(f"Provider ready: {provider.name}  (embed {provider.embed_model})")

# %% ── CELL 3: Load comments ───────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)
if TEXT_COL not in df.columns:
    raise SystemExit(f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}")
df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
if SAMPLE_SIZE:
    df = df.head(SAMPLE_SIZE)

id_ok = ID_COL and ID_COL in df.columns
comments = [
    {"comment_id": int(df[ID_COL].iloc[i]) if id_ok else i + 1,
     "text":       str(df[TEXT_COL].iloc[i])}
    for i in range(len(df))
]
print(f"Loaded {len(comments)} comments.")
print(f"  Example: {comments[0]['text'][:120]}")

# %% ── CELL 4: Sentence chunking (deterministic, verbatim) ─────────────────────
# Splits on sentence boundaries while protecting decimals and common abbreviations,
# then merges short sentences forward so no chunk is a context-free fragment.

_ABBREV = r'\b(?:Dr|Mr|Mrs|Ms|vs|etc|e\.g|i\.e|U\.S|a\.m|p\.m|St|approx|No|Inc|Co|Rd|Ave)\.'

def split_sentences(text: str) -> list:
    t = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)                 # protect 3.5, $4.20
    t = re.sub(_ABBREV, lambda m: m.group(0).replace('.', '<DOT>'), t, flags=re.IGNORECASE)
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.replace('<DOT>', '.').strip() for p in parts if p.strip()]

def chunk_text(text: str, min_words: int) -> list:
    sents, chunks, buf = split_sentences(text), [], ""
    for s in sents:
        buf = f"{buf} {s}".strip() if buf else s
        if len(buf.split()) >= min_words:
            chunks.append(buf); buf = ""
    if buf:                                  # attach trailing remainder for context
        if chunks: chunks[-1] = f"{chunks[-1]} {buf}"
        else:      chunks.append(buf)
    return chunks

chunks = []
for c in comments:
    for j, ck in enumerate(chunk_text(c["text"], MIN_CHUNK_WORDS)):
        chunks.append({
            "doc_id":    len(chunks) + 1,    # unique chunk id (used by search funcs)
            "parent_id": c["comment_id"],    # the whole comment this came from
            "text":      ck,
        })

per_comment = [len(chunk_text(c["text"], MIN_CHUNK_WORDS)) for c in comments]
print(f"{len(comments)} comments -> {len(chunks)} chunks "
      f"(avg {np.mean(per_comment):.2f}, max {max(per_comment)} per comment)")
multi = sum(1 for n in per_comment if n > 1)
print(f"  Multi-chunk comments: {multi}/{len(comments)} ({multi/len(comments):.0%})  "
      f"<- higher % means chunking is more likely to help")
print(f"  Example chunks of comment 1:")
for ck in [c for c in chunks if c['parent_id'] == comments[0]['comment_id']]:
    print(f"    - {ck['text'][:90]}")

# %% ── CELL 5: Embed chunks (batched) ──────────────────────────────────────────

texts = [c["text"] for c in chunks]
print(f"Embedding {len(texts)} chunks in batches of {EMBED_BATCH} ...")
vecs = []
for i in range(0, len(texts), EMBED_BATCH):
    batch = texts[i:i + EMBED_BATCH]
    try:
        vecs.extend(provider.embed_texts(batch, task_type="retrieval_document"))
    except Exception as e:
        print(f"  [WARN] batch {i} failed ({type(e).__name__}) — falling back per-item")
        vecs.extend(provider.embed_text(t, task_type="retrieval_document") for t in batch)
    print(f"  {min(i + EMBED_BATCH, len(texts))}/{len(texts)}")
    time.sleep(2)
matrix = np.array(vecs, dtype=np.float32)
print(f"Done — matrix {matrix.shape}")

# %% ── CELL 6: Save chunk cache ────────────────────────────────────────────────
# documents.json holds the CHUNKS (so calibrate_thresholds.py works if you point
# its CACHE_DIR at ./chunk_cache); comments.json holds whole comments for rollup.

cache = Path(CHUNK_CACHE_DIR); cache.mkdir(parents=True, exist_ok=True)
tag   = provider.embed_model.replace("/", "_").replace(".", "_").replace("-", "_").replace(":", "_")
npy   = cache / f"chunks_{provider.name.lower()}_{tag}.npy"

np.save(str(npy), matrix)
json.dump(chunks,   open(cache / "documents.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
json.dump(comments, open(cache / "comments.json",  "w", encoding="utf-8"), indent=2, ensure_ascii=False)
json.dump({
    "doc_count":   len(chunks),
    "comments":    len(comments),
    "csv_path":    str(Path(CSV_PATH).resolve()),
    "embed_model": provider.embed_model,
    "kind":        "sentence-chunks",
    "min_chunk_words": MIN_CHUNK_WORDS,
}, open(npy.with_suffix(".meta.json"), "w", encoding="utf-8"), indent=2)
print(f"Saved -> {npy.name}  +  documents.json (chunks)  +  comments.json")

# %% ── CELL 7: TEST SEARCH over chunks, rolled up to whole comments ────────────
# Re-run this cell with different QUERY values. It runs the full search_v2 pipeline
# on CHUNKS, then collapses matching chunks back to unique comments.
#
# NOTE: thresholds are inherited from search_v2 but were calibrated on WHOLE
# comments. Chunks have a different score distribution — point
# calibrate_thresholds.py at ./chunk_cache to recalibrate for a fair comparison.

QUERY = "prior auth"

chunk_docs = json.load(open(cache / "documents.json", encoding="utf-8"))
comment_by_id = {c["comment_id"]: c["text"]
                 for c in json.load(open(cache / "comments.json", encoding="utf-8"))}
raw = np.load(str(npy)).astype(np.float32)
qmean = embedding_mean(raw)
dm    = center_normalize(raw, qmean)

equivalents, variants = generate_expansions(provider, QUERY)
fuzzy_res = fuzzy_search(equivalents, chunk_docs, FUZZY_THRESHOLD)
sem_res, best = semantic_search(variants, provider, dm, chunk_docs, SEMANTIC_MIN_COSINE, query_mean=qmean)
merged, counts = rrf_merge(fuzzy_res, sem_res, best_cosines=best, documents=chunk_docs)
relevant_chunks = llm_rerank(merged, QUERY, provider)

# Roll up: best-scoring chunk per parent comment
by_parent = {}
for ch in relevant_chunks:
    pid = ch["parent_id"]
    if pid not in by_parent or ch["rrf_score"] > by_parent[pid]["rrf_score"]:
        by_parent[pid] = ch
results = sorted(by_parent.values(), key=lambda x: -x["rrf_score"])

print(f"\nQuery: '{QUERY}'")
print(f"  {counts['total']} candidate chunks -> {len(relevant_chunks)} relevant chunks "
      f"-> {len(results)} unique comments\n")
for i, ch in enumerate(results, 1):
    cos = ch.get("semantic_score", 0.0)
    print(f"#{i}  comment {ch['parent_id']}  (cosine {cos:.2f}, matched chunk below)")
    print(f"    MATCHED : {ch['text'][:110]}")
    print(f"    COMMENT : {comment_by_id[ch['parent_id']][:140]}")
    print()
