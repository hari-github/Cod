# Databricks notebook source
# Comment Intelligence Pipeline
# Topic Tagging → Tag Consolidation → Sentiment Analysis → HTML Dashboard
# Model: databricks/claude-3-7-sonnet

# ============================================================
# CELL 1 — Imports & Configuration
# ============================================================

import json
import re
import time
import textwrap
from collections import defaultdict
from datetime import datetime
from itertools import islice

import pandas as pd
import requests
from IPython.display import HTML, display

# ── Databricks model serving endpoint ──────────────────────
DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"   # ← replace
DATABRICKS_TOKEN = dbutils.secrets.get("scope", "key")               # ← replace with your secret scope/key
MODEL_ENDPOINT   = "databricks-claude-3-7-sonnet"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

# ── Tuneable constants ──────────────────────────────────────
BATCH_SIZE          = 3     # comments per LLM call
MAX_RETRIES         = 3
RETRY_DELAY_SEC     = 2

print("✅ Cell 1 complete — imports & config loaded")

# ============================================================
# CELL 2 — Seed Data: Topic Taxonomy & Sample Tags
# ============================================================

# ── 10+ seed topics with descriptions ──────────────────────
TOPIC_TAXONOMY = {
    "Product Quality":         "Comments about the physical quality, durability, materials, or craftsmanship of the product.",
    "Customer Service":        "Interactions with support teams, responsiveness, helpfulness, or resolution quality.",
    "Delivery & Shipping":     "Speed, packaging, tracking accuracy, or damage during transit.",
    "Pricing & Value":         "Cost perception, affordability, price-to-quality ratio, discounts, or hidden fees.",
    "Ease of Use / UX":        "How intuitive, simple, or frustrating the product or service experience is.",
    "Onboarding & Setup":      "First-time experience, installation, account creation, or initial configuration.",
    "Feature Requests":        "Suggestions or desires for new capabilities, improvements, or missing functionality.",
    "Reliability & Bugs":      "Software glitches, downtime, crashes, data loss, or inconsistent behavior.",
    "Communication & Updates": "Clarity of notifications, status updates, emails, or change announcements.",
    "Brand & Trust":           "Overall brand perception, reputation, transparency, or ethical concerns.",
    "Performance & Speed":     "Latency, load times, processing speed, or system responsiveness.",
    "Documentation & Help":    "Quality of FAQs, manuals, tutorials, help articles, or in-app guidance.",
    "Account & Billing":       "Subscription management, invoicing errors, refund policies, or account access issues.",
    "Personalization":         "Degree to which the product adapts to individual preferences or needs.",
}

# ── Common seed tags (LLM may extend these) ──────────────────
SEED_TAGS = [
    "fast delivery", "slow response time", "easy setup", "unclear instructions",
    "great value", "overpriced", "frequent crashes", "intuitive interface",
    "poor packaging", "excellent support", "hidden fees", "missing features",
    "buggy mobile app", "seamless onboarding", "confusing navigation",
    "quick resolution", "delayed shipment", "product damaged", "premium feel",
    "lacks customization",
]

print("✅ Cell 2 complete — topic taxonomy & seed tags defined")
print(f"   Topics defined : {len(TOPIC_TAXONOMY)}")
print(f"   Seed tags      : {len(SEED_TAGS)}")

# ============================================================
# CELL 3 — Sample DataFrame (replace with your actual df)
# ============================================================

# ── Replace this block with your real Spark/Pandas df ───────
data = {
    "SURVEY_MONTH": [
        "2025-03", "2025-03", "2025-03", "2025-03", "2025-03",
        "2025-03", "2025-04", "2025-04", "2025-04", "2025-04",
        "2025-04", "2025-04",
    ],
    "RESPONSE_ID": [f"R{str(i).zfill(3)}" for i in range(1, 13)],
    "COMMENTS": [
        "The product arrived two days late and the box was completely crushed. Inside thankfully it was fine but very disappointing packaging.",
        "Support was incredibly responsive — resolved my billing issue in under an hour. The team was friendly and professional.",
        "Setup was a nightmare. Spent three hours just trying to get the app to recognize my account. The help docs are outdated.",
        "Great value for money, especially given the premium build quality. Feels solid and looks beautiful.",
        "The mobile app crashes every time I try to export data. This has been happening for weeks and no fix in sight.",
        "I love the personalization options — it adapts perfectly to my workflow. Minor gripe: the onboarding tutorial skips too many steps.",
        "Shipping was blazing fast! Received my order in 24 hours. Packaging was neat and everything arrived intact.",
        "Customer service kept me on hold for 45 minutes and then couldn't resolve my refund request. Very frustrating.",
        "The new dashboard update is clean and intuitive. Performance has improved dramatically — pages load almost instantly.",
        "I was charged twice for the same subscription month. Took two weeks of back-and-forth to get a refund. Unacceptable.",
        "The product quality has noticeably declined compared to a year ago. Materials feel cheap and the finish is scratchy.",
        "Documentation for the API is excellent — very thorough with real-world examples. Made integration a breeze.",
    ],
}

df = pd.DataFrame(data)

# ── Identify current and prior month ────────────────────────
all_months   = sorted(df["SURVEY_MONTH"].unique())
current_month = all_months[-1]
prior_month   = all_months[-2] if len(all_months) > 1 else None

df_current = df[df["SURVEY_MONTH"] == current_month].reset_index(drop=True)
df_prior   = df[df["SURVEY_MONTH"] == prior_month].reset_index(drop=True) if prior_month else pd.DataFrame()

print(f"✅ Cell 3 complete — dataframe loaded")
print(f"   Current month  : {current_month}  ({len(df_current)} comments)")
print(f"   Prior month    : {prior_month}  ({len(df_prior)} comments)" if prior_month else "   Prior month    : None")

# ============================================================
# CELL 4 — LLM Helper
# ============================================================

def call_llm(system_prompt: str, user_prompt: str, label: str = "") -> str:
    """
    Single call to Databricks Claude endpoint.
    Returns the raw text content string.
    """
    url     = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL_ENDPOINT}/invocations"
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.2,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return content.strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"[{label}] LLM call failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_DELAY_SEC * attempt)


def extract_json(raw: str) -> dict | list:
    """
    Strip markdown fences and parse JSON safely.
    """
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Grab first balanced JSON object or array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"No JSON found in:\n{raw[:300]}")


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


print("✅ Cell 4 complete — LLM helpers defined")

# ============================================================
# CELL 5 — Stage 1: Tag Extraction (3 comments per call)
# ============================================================

TAG_EXTRACTION_SYSTEM = textwrap.dedent("""
You are a precise survey analytics assistant. Your task is to extract granular topic tags from customer comments.

Follow this reasoning chain silently before answering:
  1. Read each comment carefully for its main complaints, praises, or observations.
  2. For each distinct idea, formulate a 2–4 word descriptive tag (noun-phrase style).
  3. Choose tags that are specific enough to be actionable (e.g., "slow export speed" not just "slow").
  4. You may use the seed tags provided but MUST generate new tags when the seed list doesn't cover the idea.
  5. Do not repeat near-duplicate tags within the same comment.

Respond ONLY with valid JSON. No prose, no markdown fences.
Format:
{
  "RESPONSE_ID": ["tag1", "tag2", ...],
  ...
}
""").strip()

def build_tag_extraction_prompt(batch: list[dict], seed_tags: list[str]) -> str:
    seed_str = ", ".join(seed_tags)
    comments_block = "\n".join(
        f'  "{row["RESPONSE_ID"]}": "{row["COMMENTS"]}"'
        for row in batch
    )
    return textwrap.dedent(f"""
Seed tags for reference (extend freely):
{seed_str}

Comments to tag (3 comments):
{{
{comments_block}
}}

Return JSON with RESPONSE_ID as keys and a list of extracted tags as values.
""").strip()


def extract_tags_for_dataframe(df_input: pd.DataFrame) -> dict:
    """Run tag extraction over all comments in batches of 3."""
    records    = df_input.to_dict("records")
    all_tags   = {}
    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, batch in enumerate(batched(records, BATCH_SIZE), 1):
        print(f"  🔖 Tagging batch {i}/{total_batches} ({[r['RESPONSE_ID'] for r in batch]})")
        prompt = build_tag_extraction_prompt(batch, SEED_TAGS)
        raw    = call_llm(TAG_EXTRACTION_SYSTEM, prompt, label=f"tag-batch-{i}")
        result = extract_json(raw)
        all_tags.update(result)

    return all_tags  # { RESPONSE_ID: [tag, ...] }


print("🔄 Stage 1 — Extracting tags for CURRENT month …")
current_tags = extract_tags_for_dataframe(df_current)
print(f"✅ Cell 5 complete — tags extracted for {len(current_tags)} comments")

if not df_prior.empty:
    print("🔄 Stage 1 — Extracting tags for PRIOR month …")
    prior_tags = extract_tags_for_dataframe(df_prior)
    print(f"   Prior month tags extracted for {len(prior_tags)} comments")
else:
    prior_tags = {}
    print("   No prior month data — skipping")

# ============================================================
# CELL 6 — Stage 2: Tag Consolidation → Topic Mapping
# ============================================================

TOPIC_MAPPING_SYSTEM = textwrap.dedent("""
You are a senior customer insights analyst. Given a flat list of extracted comment tags, your job is to map every tag to exactly one topic from a provided taxonomy.

Reason through this chain before answering:
  1. Review all unique tags.
  2. For each tag, identify which topic from the taxonomy best captures its meaning.
  3. If a tag spans two topics, assign it to the more dominant one.
  4. Every tag must be assigned — do not drop any.

Respond ONLY with valid JSON. No prose, no markdown fences.
Format:
{
  "Topic Name": ["tag1", "tag2", ...],
  ...
}
Only include topics that have at least one tag assigned.
""").strip()


def build_topic_mapping_prompt(all_tags_flat: list[str], taxonomy: dict) -> str:
    taxonomy_str = "\n".join(f'  "{t}": "{d}"' for t, d in taxonomy.items())
    tags_str     = json.dumps(sorted(set(all_tags_flat)), indent=2)
    return textwrap.dedent(f"""
Topic taxonomy (name → description):
{{
{taxonomy_str}
}}

All unique tags extracted from comments:
{tags_str}

Map every tag to its most relevant topic. Return JSON.
""").strip()


def consolidate_tags_to_topics(tags_by_comment: dict, taxonomy: dict) -> dict:
    """Flatten all tags, then map to topics in one LLM call."""
    all_tags_flat = [tag for tags in tags_by_comment.values() for tag in tags]
    prompt        = build_topic_mapping_prompt(all_tags_flat, taxonomy)
    raw           = call_llm(TOPIC_MAPPING_SYSTEM, prompt, label="topic-mapping")
    return extract_json(raw)  # { Topic: [tag, ...] }


# Build a lookup: tag → topic
def build_tag_topic_lookup(topic_to_tags: dict) -> dict:
    return {tag: topic for topic, tags in topic_to_tags.items() for tag in tags}


print("🔄 Stage 2 — Consolidating tags → topics for CURRENT month …")
current_topic_to_tags   = consolidate_tags_to_topics(current_tags, TOPIC_TAXONOMY)
current_tag_topic_lookup = build_tag_topic_lookup(current_topic_to_tags)

if prior_tags:
    print("🔄 Stage 2 — Consolidating tags → topics for PRIOR month …")
    prior_topic_to_tags    = consolidate_tags_to_topics(prior_tags, TOPIC_TAXONOMY)
    prior_tag_topic_lookup = build_tag_topic_lookup(prior_topic_to_tags)
else:
    prior_topic_to_tags    = {}
    prior_tag_topic_lookup = {}

print("✅ Cell 6 complete — topic mapping done")
print(f"   Topics active (current) : {list(current_topic_to_tags.keys())}")

# ============================================================
# CELL 7 — Derive Topics Per Comment
# ============================================================

def derive_comment_topics(tags_by_comment: dict, tag_topic_lookup: dict) -> dict:
    """
    For each comment, map its tags to topics.
    Returns: { RESPONSE_ID: [topic, ...] } (deduplicated, ordered)
    """
    result = {}
    for resp_id, tags in tags_by_comment.items():
        seen   = []
        seen_s = set()
        for tag in tags:
            topic = tag_topic_lookup.get(tag)
            if topic and topic not in seen_s:
                seen.append(topic)
                seen_s.add(topic)
        result[resp_id] = seen
    return result


current_comment_topics = derive_comment_topics(current_tags, current_tag_topic_lookup)
prior_comment_topics   = derive_comment_topics(prior_tags, prior_tag_topic_lookup) if prior_tags else {}

print("✅ Cell 7 complete — topics derived per comment")
for rid, topics in list(current_comment_topics.items())[:3]:
    print(f"   {rid}: {topics}")

# ============================================================
# CELL 8 — Stage 3: Sentiment Analysis (3 comments per call)
# ============================================================

SENTIMENT_SYSTEM = textwrap.dedent("""
You are a meticulous sentiment analysis specialist. For each comment you receive, you will identify sentiment for EACH topic present in that comment independently.

Reason through this chain silently before answering:
  1. Read the full comment.
  2. For each assigned topic, locate the relevant sentences or phrases.
  3. Classify sentiment as one of: "positive", "negative", or "neutral".
  4. A single comment can have mixed sentiments across different topics.
  5. Only assign sentiment to the topics listed for that comment — do not add new topics.

Respond ONLY with valid JSON. No prose, no markdown fences.
Format:
{
  "RESPONSE_ID": {
    "Topic Name": "positive" | "negative" | "neutral",
    ...
  },
  ...
}
""").strip()


def build_sentiment_prompt(batch: list[dict], comment_topics: dict) -> str:
    entries = []
    for row in batch:
        rid    = row["RESPONSE_ID"]
        topics = comment_topics.get(rid, [])
        entries.append(
            f'  "{rid}": {{\n    "comment": "{row["COMMENTS"].replace(chr(34), chr(39))}",\n    "topics": {json.dumps(topics)}\n  }}'
        )
    body = ",\n".join(entries)
    return textwrap.dedent(f"""
Comments with their assigned topics:
{{
{body}
}}

For each comment, return the sentiment for each of its listed topics.
""").strip()


def analyze_sentiment(df_input: pd.DataFrame, comment_topics: dict) -> dict:
    """Run sentiment analysis in batches of 3."""
    records        = df_input.to_dict("records")
    all_sentiments = {}
    total_batches  = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE

    for i, batch in enumerate(batched(records, BATCH_SIZE), 1):
        print(f"  💬 Sentiment batch {i}/{total_batches}")
        prompt = build_sentiment_prompt(batch, comment_topics)
        raw    = call_llm(SENTIMENT_SYSTEM, prompt, label=f"sentiment-batch-{i}")
        result = extract_json(raw)
        all_sentiments.update(result)

    return all_sentiments  # { RESPONSE_ID: { Topic: sentiment } }


print("🔄 Stage 3 — Sentiment analysis for CURRENT month …")
current_sentiment = analyze_sentiment(df_current, current_comment_topics)

if not df_prior.empty:
    print("🔄 Stage 3 — Sentiment analysis for PRIOR month …")
    prior_sentiment = analyze_sentiment(df_prior, prior_comment_topics)
else:
    prior_sentiment = {}

print("✅ Cell 8 complete — sentiment analysis done")

# ============================================================
# CELL 9 — Stage 4: Topic-Level Summary & Commentary
# ============================================================

SUMMARY_SYSTEM = textwrap.dedent("""
You are a strategic insights writer. You will receive grouped comments for a single topic — split by sentiment — and produce:
  1. A 2–3 sentence summary for "positive" comments (if any).
  2. A 2–3 sentence summary for "negative" comments (if any).
  3. A 1–2 sentence "neutral" summary (if any).
  4. A 2–3 sentence comparative commentary comparing current month vs prior month sentiment distribution.

Reason silently:
  - Identify dominant themes in each sentiment group.
  - Use specific, concrete language from the comments.
  - Make the commentary actionable and directional.

Respond ONLY with valid JSON. No prose, no markdown fences.
Format:
{
  "positive_summary": "...",
  "negative_summary": "...",
  "neutral_summary": "...",
  "commentary": "..."
}
Omit keys for sentiment groups with no comments. Always include commentary if prior data exists.
""").strip()


def build_summary_prompt(
    topic: str,
    current_pos: list, current_neg: list, current_neu: list,
    prior_pos: list,   prior_neg: list,   prior_neu: list,
    has_prior: bool,
) -> str:
    def fmt(lst): return "\n".join(f"  - {c}" for c in lst) if lst else "  (none)"
    prior_block = ""
    if has_prior:
        prior_block = f"""
PRIOR MONTH comments:
  Positive ({len(prior_pos)}):
{fmt(prior_pos)}
  Negative ({len(prior_neg)}):
{fmt(prior_neg)}
  Neutral ({len(prior_neu)}):
{fmt(prior_neu)}
"""
    return textwrap.dedent(f"""
Topic: {topic}

CURRENT MONTH comments:
  Positive ({len(current_pos)}):
{fmt(current_pos)}
  Negative ({len(current_neg)}):
{fmt(current_neg)}
  Neutral ({len(current_neu)}):
{fmt(current_neu)}
{prior_block}
Generate summaries and commentary as instructed.
""").strip()


def build_topic_sentiments(df_input, sentiment_map):
    """Returns { topic: { 'positive': [comments], 'negative': [...], 'neutral': [...] } }"""
    lookup = df_input.set_index("RESPONSE_ID")["COMMENTS"].to_dict()
    result = defaultdict(lambda: {"positive": [], "negative": [], "neutral": []})
    for rid, topic_sentiments in sentiment_map.items():
        comment_text = lookup.get(rid, "")
        for topic, sent in topic_sentiments.items():
            result[topic][sent].append(comment_text)
    return result


print("🔄 Stage 4 — Generating topic summaries & commentary …")

current_topic_sents = build_topic_sentiments(df_current, current_sentiment)
prior_topic_sents   = build_topic_sentiments(df_prior, prior_sentiment) if not df_prior.empty else {}

topic_summaries = {}
all_topics = sorted(current_topic_sents.keys())

for topic in all_topics:
    print(f"  📝 Summarising: {topic}")
    cs = current_topic_sents[topic]
    ps = prior_topic_sents.get(topic, {"positive": [], "negative": [], "neutral": []})
    has_prior = bool(prior_topic_sents)

    prompt = build_summary_prompt(
        topic,
        cs["positive"], cs["negative"], cs["neutral"],
        ps["positive"], ps["negative"], ps["neutral"],
        has_prior,
    )
    raw    = call_llm(SUMMARY_SYSTEM, prompt, label=f"summary-{topic}")
    result = extract_json(raw)
    topic_summaries[topic] = {
        "summary": result,
        "current_counts": {k: len(v) for k, v in cs.items()},
        "prior_counts":   {k: len(v) for k, v in ps.items()} if has_prior else None,
    }

print(f"✅ Cell 9 complete — summaries generated for {len(topic_summaries)} topics")

# ============================================================
# CELL 10 — Aggregation: Tag Frequency & Sentiment Matrices
# ============================================================

def compute_tag_frequencies(tags_by_comment: dict) -> dict:
    freq = defaultdict(int)
    for tags in tags_by_comment.values():
        for tag in tags:
            freq[tag] += 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))


def compute_topic_sentiment_matrix(sentiment_map: dict) -> dict:
    """{ topic: { positive: n, negative: n, neutral: n } }"""
    matrix = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for topic_sents in sentiment_map.values():
        for topic, sent in topic_sents.items():
            matrix[topic][sent] += 1
    return {k: dict(v) for k, v in matrix.items()}


current_tag_freq = compute_tag_frequencies(current_tags)
prior_tag_freq   = compute_tag_frequencies(prior_tags) if prior_tags else {}

current_sent_matrix = compute_topic_sentiment_matrix(current_sentiment)
prior_sent_matrix   = compute_topic_sentiment_matrix(prior_sentiment) if prior_sentiment else {}

print("✅ Cell 10 complete — frequency & sentiment matrices built")
print(f"   Top 5 tags (current): {list(current_tag_freq.items())[:5]}")

# ============================================================
# CELL 11 — Build Interactive HTML Dashboard
# ============================================================

def render_dashboard(
    current_month,
    prior_month,
    topic_summaries,
    current_sent_matrix,
    prior_sent_matrix,
    current_tag_freq,
    prior_tag_freq,
    current_tags,
    current_comment_topics,
    current_topic_to_tags,
):
    # ── Serialise Python data to JS ──────────────────────────
    topics        = sorted(topic_summaries.keys())
    topic_js      = json.dumps(topics)
    summaries_js  = json.dumps(topic_summaries)
    sent_cur_js   = json.dumps(current_sent_matrix)
    sent_pri_js   = json.dumps(prior_sent_matrix)
    tag_cur_js    = json.dumps(dict(list(current_tag_freq.items())[:40]))
    tag_pri_js    = json.dumps(dict(list(prior_tag_freq.items())[:40]))
    topic_tags_js = json.dumps(current_topic_to_tags)
    has_prior     = "true" if prior_month else "false"
    cur_label     = current_month
    pri_label     = prior_month or ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Comment Intelligence Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:        #0d0f14;
    --surface:   #161922;
    --surface2:  #1e2330;
    --border:    #2a3045;
    --accent:    #5b8cff;
    --accent2:   #a78bfa;
    --pos:       #34d399;
    --neg:       #f87171;
    --neu:       #94a3b8;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --gold:      #fbbf24;
    --r:         12px;
  }}
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 0;
  }}

  /* ── Header ── */
  .header {{
    background: linear-gradient(135deg, #0d0f14 0%, #161922 50%, #1a1040 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 48px 24px;
    position: relative;
    overflow: hidden;
  }}
  .header::before {{
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(91,140,255,0.08) 0%, transparent 70%);
    pointer-events: none;
  }}
  .header-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #e2e8f0, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .header-meta {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 6px;
    letter-spacing: 0.5px;
  }}
  .period-badges {{
    display: flex; gap: 8px; margin-top: 16px;
  }}
  .badge {{
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .badge-cur {{ border-color: var(--accent); color: var(--accent); background: rgba(91,140,255,0.08); }}
  .badge-pri {{ border-color: var(--accent2); color: var(--accent2); background: rgba(167,139,250,0.08); }}

  /* ── Layout ── */
  .main {{ padding: 32px 48px; display: flex; flex-direction: column; gap: 28px; }}
  .section-title {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--text);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
  }}
  .section-title::after {{
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }}

  /* ── Stat cards ── */
  .stats-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }}
  .stat-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 20px;
    position: relative;
    overflow: hidden;
  }}
  .stat-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
  }}
  .stat-label {{ font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }}
  .stat-value {{
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    margin-top: 4px;
    line-height: 1;
  }}
  .stat-sub {{ font-size: 0.72rem; color: var(--muted); margin-top: 6px; }}

  /* ── Tab navigation ── */
  .tabs {{ display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }}
  .tab {{
    padding: 10px 20px;
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
    border-radius: 4px 4px 0 0;
    user-select: none;
  }}
  .tab:hover {{ color: var(--text); }}
  .tab.active {{ color: var(--accent); border-bottom-color: var(--accent); }}
  .tab-content {{ display: none; }}
  .tab-content.active {{ display: block; }}

  /* ── Chart cards ── */
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 24px;
  }}
  .card-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 18px;
  }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .chart-wrap {{ position: relative; height: 260px; }}
  canvas {{ border-radius: 6px; }}

  /* ── Sentiment trend bars ── */
  .topic-filter {{
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px;
  }}
  .filter-btn {{
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.72rem; font-weight: 500;
    border: 1px solid var(--border);
    background: transparent; color: var(--muted);
    cursor: pointer; transition: all 0.15s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    border-color: var(--accent); color: var(--accent);
    background: rgba(91,140,255,0.08);
  }}

  /* ── Topic sentiment bars ── */
  .topic-bars {{ display: flex; flex-direction: column; gap: 12px; }}
  .topic-row {{ display: grid; grid-template-columns: 160px 1fr 80px; align-items: center; gap: 12px; }}
  .topic-name {{ font-size: 0.78rem; font-weight: 500; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .bar-group {{ display: flex; flex-direction: column; gap: 3px; }}
  .bar-track {{ background: var(--surface2); border-radius: 4px; height: 8px; overflow: hidden; position: relative; }}
  .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.6s ease; }}
  .bar-pos {{ background: var(--pos); }}
  .bar-neg {{ background: var(--neg); }}
  .bar-neu {{ background: var(--neu); }}
  .bar-label {{ font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--muted); }}
  .score-pill {{
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 8px;
    border-radius: 6px;
    text-align: center;
  }}

  /* ── Tag heatmap ── */
  .heatmap {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: flex-start; }}
  .tag-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    transition: transform 0.15s;
    cursor: default;
  }}
  .tag-chip:hover {{ transform: translateY(-1px); }}
  .tag-count {{
    background: rgba(0,0,0,0.25);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.65rem;
  }}

  /* ── Topic summary cards ── */
  .summary-grid {{ display: flex; flex-direction: column; gap: 16px; }}
  .summary-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
  }}
  .summary-header {{
    padding: 14px 20px;
    background: var(--surface2);
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    cursor: pointer; user-select: none;
  }}
  .summary-header:hover {{ background: #242a3a; }}
  .summary-topic-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
  }}
  .summary-pills {{ display: flex; gap: 6px; align-items: center; }}
  .mini-pill {{
    font-size: 0.65rem;
    font-family: 'DM Mono', monospace;
    padding: 2px 8px; border-radius: 10px;
  }}
  .pill-pos {{ background: rgba(52,211,153,0.15); color: var(--pos); border: 1px solid rgba(52,211,153,0.3); }}
  .pill-neg {{ background: rgba(248,113,113,0.15); color: var(--neg); border: 1px solid rgba(248,113,113,0.3); }}
  .pill-neu {{ background: rgba(148,163,184,0.15); color: var(--neu); border: 1px solid rgba(148,163,184,0.3); }}
  .chevron {{ color: var(--muted); font-size: 0.8rem; transition: transform 0.2s; }}
  .summary-body {{
    padding: 0 20px;
    max-height: 0; overflow: hidden;
    transition: max-height 0.35s ease, padding 0.2s ease;
  }}
  .summary-body.open {{
    padding: 18px 20px;
    max-height: 600px;
  }}
  .sentiment-block {{ margin-bottom: 14px; }}
  .sentiment-block:last-child {{ margin-bottom: 0; }}
  .sent-label {{
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;
    font-weight: 600; margin-bottom: 5px;
    display: flex; align-items: center; gap: 6px;
  }}
  .sent-dot {{ width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }}
  .sent-text {{ font-size: 0.82rem; line-height: 1.6; color: #b0bec5; }}
  .commentary-block {{
    background: rgba(91,140,255,0.06);
    border: 1px solid rgba(91,140,255,0.2);
    border-radius: 8px;
    padding: 14px;
    margin-top: 14px;
  }}
  .commentary-label {{
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;
    color: var(--accent); font-weight: 600; margin-bottom: 6px;
  }}
  .commentary-text {{ font-size: 0.82rem; line-height: 1.6; color: #a0aec0; }}

  /* ── Tooltip ── */
  .tooltip {{
    position: fixed; z-index: 999;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.75rem;
    max-width: 260px;
    pointer-events: none;
    display: none;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }}

  /* ── Legend ── */
  .legend {{ display: flex; gap: 16px; margin-bottom: 14px; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 0.72rem; color: var(--muted); }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: var(--surface); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>

<div class="tooltip" id="tooltip"></div>

<!-- HEADER -->
<div class="header">
  <div class="header-title">Comment Intelligence Dashboard</div>
  <div class="header-meta">Survey feedback analysis — AI-extracted topics, tags &amp; sentiment</div>
  <div class="period-badges">
    <span class="badge badge-cur">Current: {cur_label}</span>
    {'<span class="badge badge-pri">Prior: ' + pri_label + '</span>' if pri_label else ''}
  </div>
</div>

<div class="main">

  <!-- STAT CARDS -->
  <div id="stat-cards" class="stats-row"></div>

  <!-- TABS -->
  <div>
    <div class="tabs">
      <div class="tab active" onclick="switchTab(0)">Sentiment Trends</div>
      <div class="tab" onclick="switchTab(1)">Tag Heatmap</div>
      <div class="tab" onclick="switchTab(2)">Topic Summaries</div>
    </div>

    <!-- TAB 0: Sentiment Trends -->
    <div class="tab-content active" id="tab-0">
      <div class="charts-grid">
        <div class="card">
          <div class="card-title">Sentiment Distribution — Current Month</div>
          <div class="chart-wrap"><canvas id="chartCurPie"></canvas></div>
        </div>
        {'<div class="card"><div class="card-title">Sentiment Distribution — Prior Month</div><div class="chart-wrap"><canvas id="chartPriPie"></canvas></div></div>' if pri_label else '<div></div>'}
      </div>
      <div class="card" style="margin-top:18px">
        <div class="card-title">Topic-Level Sentiment Breakdown</div>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:var(--pos)"></div>Positive</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--neg)"></div>Negative</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--neu)"></div>Neutral</div>
        </div>
        <div class="topic-filter" id="topic-filter-btns"></div>
        <div class="topic-bars" id="topic-bars"></div>
      </div>
    </div>

    <!-- TAB 1: Tag Heatmap -->
    <div class="tab-content" id="tab-1">
      <div class="charts-grid">
        <div class="card">
          <div class="card-title">Tag Frequency — Current Month ({cur_label})</div>
          <div class="heatmap" id="heatmap-cur"></div>
        </div>
        {'<div class="card"><div class="card-title">Tag Frequency — Prior Month (' + pri_label + ')</div><div class="heatmap" id="heatmap-pri"></div></div>' if pri_label else '<div></div>'}
      </div>
    </div>

    <!-- TAB 2: Topic Summaries -->
    <div class="tab-content" id="tab-2">
      <div class="summary-grid" id="summary-grid"></div>
    </div>
  </div>
</div>

<script>
// ── Data ────────────────────────────────────────────────────
const TOPICS        = {topic_js};
const SUMMARIES     = {summaries_js};
const SENT_CUR      = {sent_cur_js};
const SENT_PRI      = {sent_pri_js};
const TAG_CUR       = {tag_cur_js};
const TAG_PRI       = {tag_pri_js};
const TOPIC_TAGS    = {topic_tags_js};
const HAS_PRIOR     = {has_prior};
const CUR_LABEL     = "{cur_label}";
const PRI_LABEL     = "{pri_label}";

// ── Colors ──────────────────────────────────────────────────
const C_POS = '#34d399', C_NEG = '#f87171', C_NEU = '#94a3b8';
const ACCENT_PALETTE = [
  '#5b8cff','#a78bfa','#fbbf24','#34d399','#f87171',
  '#38bdf8','#fb923c','#e879f9','#4ade80','#f472b6',
];

// ── Helpers ─────────────────────────────────────────────────
function getEl(id) {{ return document.getElementById(id); }}
function sentCounts(matrix) {{
  let pos=0, neg=0, neu=0;
  Object.values(matrix).forEach(t => {{
    pos += t.positive||0; neg += t.negative||0; neu += t.neutral||0;
  }});
  return {{ pos, neg, neu }};
}}
function sentScore(pos, neg, neu) {{
  const total = pos + neg + neu;
  if (!total) return 0;
  return Math.round(((pos - neg) / total) * 100);
}}
function scoreColor(s) {{
  if (s >  20) return C_POS;
  if (s < -20) return C_NEG;
  return C_NEU;
}}

// ── Stat Cards ──────────────────────────────────────────────
function buildStatCards() {{
  const cc = sentCounts(SENT_CUR);
  const pc = HAS_PRIOR ? sentCounts(SENT_PRI) : null;
  const total = cc.pos + cc.neg + cc.neu;
  const score = sentScore(cc.pos, cc.neg, cc.neu);
  const activeTopics = Object.keys(SENT_CUR).length;

  const cards = [
    {{ label: 'Total Comments', value: total, sub: CUR_LABEL }},
    {{ label: 'Net Sentiment', value: (score >= 0 ? '+' : '') + score, sub: 'score (-100 to +100)', style: `color:${{scoreColor(score)}}` }},
    {{ label: 'Positive Comments', value: cc.pos, sub: `${{Math.round(cc.pos/total*100)}}% of total`, style: `color:${{C_POS}}` }},
    {{ label: 'Topics Identified', value: activeTopics, sub: 'across current month' }},
  ];
  const el = getEl('stat-cards');
  el.innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-label">${{c.label}}</div>
      <div class="stat-value" style="${{c.style||''}}">
        ${{c.value}}
      </div>
      <div class="stat-sub">${{c.sub}}</div>
    </div>
  `).join('');
}}

// ── Pie Charts ──────────────────────────────────────────────
function buildPie(canvasId, matrix) {{
  const cc = sentCounts(matrix);
  const ctx = getEl(canvasId);
  if (!ctx) return;
  new Chart(ctx, {{
    type: 'doughnut',
    data: {{
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [{{ data: [cc.pos, cc.neg, cc.neu], backgroundColor: [C_POS, C_NEG, C_NEU], borderWidth: 0, hoverOffset: 6 }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{
        legend: {{ position: 'right', labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.label}}: ${{ctx.raw}}` }} }}
      }},
      cutout: '65%'
    }}
  }});
}}

// ── Topic Bars ──────────────────────────────────────────────
let activeTopicFilter = null;
function buildTopicFilterBtns() {{
  const el = getEl('topic-filter-btns');
  el.innerHTML = '<button class="filter-btn active" onclick="filterTopics(null, this)">All Topics</button>' +
    TOPICS.map(t => `<button class="filter-btn" onclick="filterTopics('${{t}}', this)">${{t}}</button>`).join('');
}}

function filterTopics(topic, btn) {{
  activeTopicFilter = topic;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  renderTopicBars();
}}

function renderTopicBars() {{
  const el = getEl('topic-bars');
  const topics = activeTopicFilter ? [activeTopicFilter] : TOPICS.filter(t => SENT_CUR[t]);
  el.innerHTML = topics.map(topic => {{
    const c = SENT_CUR[topic] || {{}};
    const p = SENT_PRI[topic] || {{}};
    const cp = c.positive||0, cn = c.negative||0, cne = c.neutral||0;
    const total = cp + cn + cne;
    if (!total) return '';
    const score = sentScore(cp, cn, cne);
    const pct = v => total ? Math.round(v/total*100) : 0;
    const priorInfo = HAS_PRIOR && p ? ` | Prior: +${{p.positive||0}} -${{p.negative||0}}` : '';
    return `
      <div class="topic-row">
        <div class="topic-name" title="${{topic}}">${{topic}}</div>
        <div class="bar-group">
          <div class="bar-label">+${{cp}} -${{cn}} ≈${{cne}}${{priorInfo}}</div>
          <div class="bar-track"><div class="bar-fill bar-pos" style="width:${{pct(cp)}}%"></div></div>
          <div class="bar-track"><div class="bar-fill bar-neg" style="width:${{pct(cn)}}%"></div></div>
          <div class="bar-track"><div class="bar-fill bar-neu" style="width:${{pct(cne)}}%"></div></div>
        </div>
        <div class="score-pill" style="background:${{scoreColor(score)}}22;color:${{scoreColor(score)}}">
          ${{score >= 0 ? '+' : ''}}${{score}}
        </div>
      </div>`;
  }}).join('');
}}

// ── Tag Heatmap ──────────────────────────────────────────────
function tagColor(freq, maxFreq) {{
  const ratio = freq / maxFreq;
  if (ratio > 0.75) return {{ bg: 'rgba(91,140,255,0.35)', color: '#93c5fd' }};
  if (ratio > 0.5)  return {{ bg: 'rgba(167,139,250,0.25)', color: '#c4b5fd' }};
  if (ratio > 0.25) return {{ bg: 'rgba(251,191,36,0.15)', color: '#fde68a' }};
  return {{ bg: 'rgba(100,116,139,0.15)', color: '#94a3b8' }};
}}

function buildHeatmap(containerId, tagFreq) {{
  const el = getEl(containerId);
  if (!el) return;
  const entries = Object.entries(tagFreq);
  const maxFreq = entries[0]?.[1] || 1;
  el.innerHTML = entries.map(([tag, freq]) => {{
    const c = tagColor(freq, maxFreq);
    return `<div class="tag-chip" style="background:${{c.bg}};color:${{c.color}};border:1px solid ${{c.color}}33">
      ${{tag}}<span class="tag-count">${{freq}}</span>
    </div>`;
  }}).join('');
}}

// ── Topic Summaries ──────────────────────────────────────────
function buildSummaryCards() {{
  const el = getEl('summary-grid');
  el.innerHTML = TOPICS.map((topic, i) => {{
    const data  = SUMMARIES[topic];
    if (!data) return '';
    const s = data.summary;
    const cc = data.current_counts || {{}};
    const pc = data.prior_counts;

    const pills = [
      cc.positive ? `<span class="mini-pill pill-pos">+${{cc.positive}}</span>` : '',
      cc.negative ? `<span class="mini-pill pill-neg">-${{cc.negative}}</span>` : '',
      cc.neutral  ? `<span class="mini-pill pill-neu">≈${{cc.neutral}}</span>` : '',
    ].join('');

    const blocks = [];
    if (s.positive_summary)
      blocks.push(`<div class="sentiment-block">
        <div class="sent-label"><div class="sent-dot" style="background:var(--pos)"></div><span style="color:var(--pos)">Positive</span></div>
        <div class="sent-text">${{s.positive_summary}}</div>
      </div>`);
    if (s.negative_summary)
      blocks.push(`<div class="sentiment-block">
        <div class="sent-label"><div class="sent-dot" style="background:var(--neg)"></div><span style="color:var(--neg)">Negative</span></div>
        <div class="sent-text">${{s.negative_summary}}</div>
      </div>`);
    if (s.neutral_summary)
      blocks.push(`<div class="sentiment-block">
        <div class="sent-label"><div class="sent-dot" style="background:var(--neu)"></div><span style="color:var(--neu)">Neutral</span></div>
        <div class="sent-text">${{s.neutral_summary}}</div>
      </div>`);
    if (s.commentary)
      blocks.push(`<div class="commentary-block">
        <div class="commentary-label">📊 Month-over-Month Commentary</div>
        <div class="commentary-text">${{s.commentary}}</div>
      </div>`);

    return `
      <div class="summary-card">
        <div class="summary-header" onclick="toggleSummary(${{i}})">
          <div class="summary-topic-name">${{topic}}</div>
          <div class="summary-pills">
            ${{pills}}
            <span class="chevron" id="chev-${{i}}">▼</span>
          </div>
        </div>
        <div class="summary-body" id="sbody-${{i}}">
          ${{blocks.join('')}}
        </div>
      </div>`;
  }}).join('');
}}

function toggleSummary(i) {{
  const body = getEl('sbody-' + i);
  const chev = getEl('chev-' + i);
  const open = body.classList.toggle('open');
  chev.style.transform = open ? 'rotate(180deg)' : '';
}}

// ── Tab switching ────────────────────────────────────────────
function switchTab(n) {{
  document.querySelectorAll('.tab').forEach((t, i) => t.classList.toggle('active', i === n));
  document.querySelectorAll('.tab-content').forEach((c, i) => c.classList.toggle('active', i === n));
}}

// ── Boot ─────────────────────────────────────────────────────
buildStatCards();
buildPie('chartCurPie', SENT_CUR);
if (HAS_PRIOR) buildPie('chartPriPie', SENT_PRI);
buildTopicFilterBtns();
renderTopicBars();
buildHeatmap('heatmap-cur', TAG_CUR);
if (HAS_PRIOR) buildHeatmap('heatmap-pri', TAG_PRI);
buildSummaryCards();
// Auto-open first summary
setTimeout(() => {{ const b = getEl('sbody-0'); if(b) {{ b.classList.add('open'); const c=getEl('chev-0'); if(c) c.style.transform='rotate(180deg)'; }} }}, 400);
</script>
</body>
</html>"""
    return html


html_output = render_dashboard(
    current_month       = current_month,
    prior_month         = prior_month,
    topic_summaries     = topic_summaries,
    current_sent_matrix = current_sent_matrix,
    prior_sent_matrix   = prior_sent_matrix,
    current_tag_freq    = current_tag_freq,
    prior_tag_freq      = prior_tag_freq,
    current_tags        = current_tags,
    current_comment_topics = current_comment_topics,
    current_topic_to_tags  = current_topic_to_tags,
)

print("✅ Cell 11 complete — HTML dashboard rendered")

# ============================================================
# CELL 12 — Display Dashboard in Notebook
# ============================================================

display(HTML(html_output))

# ============================================================
# CELL 13 — (Optional) Export to File
# ============================================================

output_path = f"/tmp/comment_dashboard_{current_month}.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_output)

print(f"✅ Cell 13 complete — dashboard saved to: {output_path}")
print("   Open the file or use dbutils.fs / display(HTML(...)) to view it.")
