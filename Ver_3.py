# Databricks Notebook Source
# Comment Intelligence Pipeline v2
# ─────────────────────────────────────────────────────────────
# Stage 1 : Tag Extraction          (3 comments / LLM call)
# Stage 2 : Tag → Topic Mapping     (1 LLM call)
# Stage 3 : Sentiment per Topic     (3 comments / LLM call)
# Stage 4 : Topic Summary / MoM     (1 LLM call / topic)
# Output  : Interactive HTML Dashboard + persisted JSON
# ─────────────────────────────────────────────────────────────


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 1 — Imports & Configuration                           ║
# ╚══════════════════════════════════════════════════════════════╝

import json, re, os, time, textwrap
from collections import defaultdict
from itertools import islice

import pandas as pd
import requests
from IPython.display import display, HTML

# ── Databricks endpoint ────────────────────────────────────────
DATABRICKS_HOST  = "https://<your-workspace>.azuredatabricks.net"  # ← fill in
DATABRICKS_TOKEN = dbutils.secrets.get("scope", "key")             # ← fill in
MODEL_ENDPOINT   = "databricks-claude-3-7-sonnet"

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type":  "application/json",
}

# ── Persistence root ───────────────────────────────────────────
# Files saved as: {RESULTS_ROOT}/survey_results_YYYY-MM.json
RESULTS_ROOT = "/Workspace/Users/<your-user>/survey_pipeline"      # ← fill in

os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── Tuneable constants ─────────────────────────────────────────
BATCH_SIZE    = 3
MAX_RETRIES   = 3
RETRY_DELAY_S = 2

print("✅ Cell 1 — imports & config loaded")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2 — Topic Taxonomy & Seed Tags                        ║
# ╚══════════════════════════════════════════════════════════════╝

TOPIC_TAXONOMY = {
    "Product Quality":         "Physical quality, durability, materials, or craftsmanship of the product.",
    "Customer Service":        "Interactions with support teams, responsiveness, helpfulness, resolution quality.",
    "Delivery & Shipping":     "Speed, packaging, tracking accuracy, or damage during transit.",
    "Pricing & Value":         "Cost perception, affordability, price-to-quality ratio, discounts, hidden fees.",
    "Ease of Use / UX":        "How intuitive, simple, or frustrating the product/service experience is.",
    "Onboarding & Setup":      "First-time experience, installation, account creation, initial configuration.",
    "Feature Requests":        "Suggestions for new capabilities, improvements, or missing functionality.",
    "Reliability & Bugs":      "Software glitches, downtime, crashes, data loss, inconsistent behavior.",
    "Communication & Updates": "Clarity of notifications, status updates, emails, or change announcements.",
    "Brand & Trust":           "Overall brand perception, transparency, or ethical concerns.",
    "Performance & Speed":     "Latency, load times, processing speed, or system responsiveness.",
    "Documentation & Help":    "Quality of FAQs, manuals, tutorials, help articles, or in-app guidance.",
    "Account & Billing":       "Subscription management, invoicing errors, refund policies, account access.",
    "Personalization":         "Degree to which the product adapts to individual preferences or needs.",
}

SEED_TAGS = [
    "fast delivery", "slow response time", "easy setup", "unclear instructions",
    "great value", "overpriced", "frequent crashes", "intuitive interface",
    "poor packaging", "excellent support", "hidden fees", "missing features",
    "buggy mobile app", "seamless onboarding", "confusing navigation",
    "quick resolution", "delayed shipment", "product damaged", "premium feel",
    "lacks customization",
]

print(f"✅ Cell 2 — {len(TOPIC_TAXONOMY)} topics, {len(SEED_TAGS)} seed tags")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3 — Load DataFrame  (replace with your real df)       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Replace this block with your actual Spark/Pandas dataframe ─
data = {
    "SURVEY_MONTH": [
        "2025-03","2025-03","2025-03","2025-03","2025-03","2025-03",
        "2025-04","2025-04","2025-04","2025-04","2025-04","2025-04",
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

# ── Identify current & prior month ────────────────────────────
all_months    = sorted(df["SURVEY_MONTH"].unique())
current_month = all_months[-1]
prior_month   = all_months[-2] if len(all_months) > 1 else None

df_current = df[df["SURVEY_MONTH"] == current_month].reset_index(drop=True)
df_prior   = df[df["SURVEY_MONTH"] == prior_month].reset_index(drop=True) if prior_month else pd.DataFrame()

print(f"✅ Cell 3 — dataframe loaded")
print(f"   Current month : {current_month}  ({len(df_current)} comments)")
print(f"   Prior month   : {prior_month}  ({len(df_prior)} comments)" if prior_month else "   Prior month   : None")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4 — Persistence Helpers                               ║
# ╚══════════════════════════════════════════════════════════════╝

def result_path(month: str) -> str:
    return os.path.join(RESULTS_ROOT, f"survey_results_{month}.json")

def save_results(month: str, payload: dict):
    path = result_path(month)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved → {path}")

def load_results(month: str) -> dict | None:
    path = result_path(month)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def ask_reuse(month: str) -> bool:
    """
    Prompt the user interactively in the notebook to decide whether
    to reuse cached results for a given month.
    Returns True = reuse, False = re-run.
    """
    answer = input(
        f"\n⚠️  Results for '{month}' already exist at:\n"
        f"   {result_path(month)}\n\n"
        f"   Reuse cached results? (yes / no): "
    ).strip().lower()
    return answer in ("yes", "y")

print("✅ Cell 4 — persistence helpers defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5 — LLM Helpers                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def call_llm(system_prompt: str, user_prompt: str, label: str = "") -> str:
    url = f"{DATABRICKS_HOST}/serving-endpoints/{MODEL_ENDPOINT}/invocations"
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
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"[{label}] failed after {MAX_RETRIES} attempts: {e}")
            time.sleep(RETRY_DELAY_S * attempt)

def extract_json(raw: str):
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    match   = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
    if match:
        return json.loads(match.group(1))
    raise ValueError(f"No JSON in response:\n{raw[:300]}")

def batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

print("✅ Cell 5 — LLM helpers defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6 — Stage 1: Tag Extraction  (3 comments / call)      ║
# ╚══════════════════════════════════════════════════════════════╝

TAG_SYS = textwrap.dedent("""
You are a precise survey analytics assistant extracting granular topic tags from customer comments.

Think through these steps silently before writing output:
  1. Read each comment for distinct complaints, praises, or observations.
  2. Per distinct idea, form a 2–4 word noun-phrase tag (e.g. "slow export speed").
  3. Prefer specific, actionable tags over vague ones.
  4. You may extend beyond the seed tag list when needed.
  5. No near-duplicate tags within the same comment.

Output ONLY valid JSON — no prose, no markdown fences.
Schema: { "RESPONSE_ID": ["tag1", "tag2", ...], ... }
""").strip()

def tag_extraction_prompt(batch, seed_tags):
    seed_str = ", ".join(seed_tags)
    comments = "\n".join(
        f'  "{r["RESPONSE_ID"]}": "{r["COMMENTS"]}"' for r in batch
    )
    return (
        f"Seed tags (extend freely):\n{seed_str}\n\n"
        f"Comments:\n{{\n{comments}\n}}\n\n"
        "Return JSON: RESPONSE_ID → list of tags."
    )

def run_tag_extraction(df_in: pd.DataFrame) -> dict:
    records = df_in.to_dict("records")
    out     = {}
    total   = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    for i, batch in enumerate(batched(records, BATCH_SIZE), 1):
        print(f"   🔖 Tag batch {i}/{total}  {[r['RESPONSE_ID'] for r in batch]}")
        raw    = call_llm(TAG_SYS, tag_extraction_prompt(batch, SEED_TAGS), f"tags-{i}")
        out.update(extract_json(raw))
    return out   # { RESPONSE_ID: [tag, ...] }

print("✅ Cell 6 — tag extraction defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7 — Stage 2: Tag → Topic Mapping  (1 call)            ║
# ╚══════════════════════════════════════════════════════════════╝

TOPIC_MAP_SYS = textwrap.dedent("""
You are a senior customer insights analyst mapping extracted tags to a topic taxonomy.

Work through these steps silently:
  1. Review all unique tags.
  2. Assign each tag to the single best-fitting topic from the taxonomy.
  3. Every tag must be assigned — none dropped.
  4. If a tag spans two topics, pick the dominant one.

Output ONLY valid JSON — no prose, no markdown fences.
Schema: { "Topic Name": ["tag1", "tag2", ...], ... }
Only include topics that receive at least one tag.
""").strip()

def topic_map_prompt(all_tags: list, taxonomy: dict) -> str:
    tax_str  = "\n".join(f'  "{t}": "{d}"' for t, d in taxonomy.items())
    tags_str = json.dumps(sorted(set(all_tags)), indent=2)
    return (
        f"Topic taxonomy:\n{{\n{tax_str}\n}}\n\n"
        f"All unique tags:\n{tags_str}\n\n"
        "Map every tag to its topic. Return JSON."
    )

def run_topic_mapping(tags_by_comment: dict, taxonomy: dict) -> dict:
    flat   = [tag for tags in tags_by_comment.values() for tag in tags]
    raw    = call_llm(TOPIC_MAP_SYS, topic_map_prompt(flat, taxonomy), "topic-map")
    return extract_json(raw)    # { Topic: [tag, ...] }

def tag_to_topic_lookup(topic_to_tags: dict) -> dict:
    return {tag: topic for topic, tags in topic_to_tags.items() for tag in tags}

def derive_comment_topics(tags_by_comment: dict, lookup: dict) -> dict:
    result = {}
    for rid, tags in tags_by_comment.items():
        seen, seen_s = [], set()
        for tag in tags:
            tp = lookup.get(tag)
            if tp and tp not in seen_s:
                seen.append(tp); seen_s.add(tp)
        result[rid] = seen
    return result

print("✅ Cell 7 — topic mapping defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8 — Stage 3: Sentiment per Topic  (3 comments / call) ║
# ╚══════════════════════════════════════════════════════════════╝

SENT_SYS = textwrap.dedent("""
You are a meticulous sentiment analysis specialist.

For each comment, assess sentiment independently for EACH listed topic.

Think through silently:
  1. Read the full comment.
  2. Locate sentences/phrases relevant to each topic.
  3. Classify: "positive", "negative", or "neutral".
  4. One comment can have mixed sentiments across topics.
  5. Only use topics listed for that comment — do not add new ones.

Output ONLY valid JSON — no prose, no markdown fences.
Schema: { "RESPONSE_ID": { "Topic Name": "positive"|"negative"|"neutral" }, ... }
""").strip()

def sentiment_prompt(batch, comment_topics: dict) -> str:
    entries = []
    for r in batch:
        rid    = r["RESPONSE_ID"]
        topics = comment_topics.get(rid, [])
        text   = r["COMMENTS"].replace('"', "'")
        entries.append(
            f'  "{rid}": {{\n    "comment": "{text}",\n    "topics": {json.dumps(topics)}\n  }}'
        )
    return "Comments with assigned topics:\n{\n" + ",\n".join(entries) + "\n}\n\nReturn sentiment JSON."

def run_sentiment(df_in: pd.DataFrame, comment_topics: dict) -> dict:
    records = df_in.to_dict("records")
    out     = {}
    total   = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    for i, batch in enumerate(batched(records, BATCH_SIZE), 1):
        print(f"   💬 Sentiment batch {i}/{total}")
        raw = call_llm(SENT_SYS, sentiment_prompt(batch, comment_topics), f"sent-{i}")
        out.update(extract_json(raw))
    return out   # { RESPONSE_ID: { Topic: sentiment } }

print("✅ Cell 8 — sentiment analysis defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 9 — Stage 4: Topic Commentary  (1 call / topic)       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Commentary system prompts ──────────────────────────────────
COMMENTARY_SYS_NO_PRIOR = textwrap.dedent("""
You are a strategic customer insights writer producing concise topic-level commentary.

Think through silently:
  1. What do positive comments most commonly say about this topic?
  2. What do negative comments most commonly say about this topic?
  3. Are there any standout observations worth highlighting?
  4. Write clear, specific, actionable sentences — avoid generic phrasing.

Output ONLY valid JSON — no prose, no markdown fences.
Schema:
{
  "positive_summary": "2-3 sentence summary of positive observations.",
  "negative_summary": "2-3 sentence summary of negative observations.",
  "neutral_summary":  "1-2 sentence summary of neutral observations (omit key if none).",
  "commentary":       "2-3 sentence summary of overall observations for this topic."
}
""").strip()

COMMENTARY_SYS_WITH_PRIOR = textwrap.dedent("""
You are a strategic customer insights writer producing topic-level commentary with month-over-month comparison.

Think through silently:
  1. Summarise what positive/negative/neutral comments say in the current month.
  2. Compare current month sentiment distribution to prior month.
  3. Identify what improved, worsened, or remained the same.
  4. Make the MoM commentary directional and specific — name the shift.

Output ONLY valid JSON — no prose, no markdown fences.
Schema:
{
  "positive_summary": "2-3 sentence summary of current positive observations.",
  "negative_summary": "2-3 sentence summary of current negative observations.",
  "neutral_summary":  "1-2 sentence summary (omit key if none).",
  "commentary":       "2-3 sentence current-month overall summary.",
  "mom_commentary":   "2-3 sentence MoM comparison: what shifted and why it matters."
}
""").strip()

def build_commentary_prompt(topic, cur_pos, cur_neg, cur_neu, pri_pos, pri_neg, pri_neu, has_prior):
    def fmt(lst): return "\n".join(f"  - {c}" for c in lst) if lst else "  (none)"
    body = (
        f"Topic: {topic}\n\n"
        f"CURRENT MONTH:\n"
        f"  Positive ({len(cur_pos)}):\n{fmt(cur_pos)}\n"
        f"  Negative ({len(cur_neg)}):\n{fmt(cur_neg)}\n"
        f"  Neutral  ({len(cur_neu)}):\n{fmt(cur_neu)}\n"
    )
    if has_prior:
        body += (
            f"\nPRIOR MONTH:\n"
            f"  Positive ({len(pri_pos)}):\n{fmt(pri_pos)}\n"
            f"  Negative ({len(pri_neg)}):\n{fmt(pri_neg)}\n"
            f"  Neutral  ({len(pri_neu)}):\n{fmt(pri_neu)}\n"
        )
    body += "\nGenerate the commentary JSON."
    return body

def group_comments_by_sentiment(df_in, sentiment_map):
    lookup = df_in.set_index("RESPONSE_ID")["COMMENTS"].to_dict()
    result = defaultdict(lambda: {"positive": [], "negative": [], "neutral": []})
    for rid, topic_sents in sentiment_map.items():
        txt = lookup.get(rid, "")
        for topic, sent in topic_sents.items():
            result[topic][sent].append(txt)
    return result

def run_commentary(
    current_grouped, prior_grouped, has_prior
) -> dict:
    summaries = {}
    topics    = sorted(current_grouped.keys())
    for topic in topics:
        print(f"   📝 Commentary: {topic}")
        cs = current_grouped[topic]
        ps = prior_grouped.get(topic, {"positive": [], "negative": [], "neutral": []}) if has_prior else {}
        sys_prompt = COMMENTARY_SYS_WITH_PRIOR if has_prior else COMMENTARY_SYS_NO_PRIOR
        prompt     = build_commentary_prompt(
            topic,
            cs["positive"], cs["negative"], cs["neutral"],
            ps.get("positive", []), ps.get("negative", []), ps.get("neutral", []),
            has_prior
        )
        raw    = call_llm(sys_prompt, prompt, f"commentary-{topic}")
        result = extract_json(raw)
        summaries[topic] = {
            "summary": result,
            "current_counts": {k: len(v) for k, v in cs.items()},
            "prior_counts":   {k: len(v) for k, v in ps.items()} if has_prior else None,
        }
    return summaries

print("✅ Cell 9 — topic commentary defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10 — Aggregation Helpers                              ║
# ╚══════════════════════════════════════════════════════════════╝

def tag_frequencies(tags_by_comment: dict) -> dict:
    freq = defaultdict(int)
    for tags in tags_by_comment.values():
        for tag in tags:
            freq[tag] += 1
    return dict(sorted(freq.items(), key=lambda x: -x[1]))

def topic_sentiment_matrix(sentiment_map: dict) -> dict:
    matrix = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})
    for topic_sents in sentiment_map.values():
        for topic, sent in topic_sents.items():
            matrix[topic][sent] += 1
    return {k: dict(v) for k, v in matrix.items()}

print("✅ Cell 10 — aggregation helpers defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 11 — Orchestrator: Run or Reuse                       ║
# ╚══════════════════════════════════════════════════════════════╝

def run_pipeline_for_month(month: str, df_in: pd.DataFrame) -> dict:
    """Run all 4 LLM stages and return a serialisable result dict."""
    print(f"\n🚀 Running pipeline for {month} ({len(df_in)} comments) …")

    # Stage 1
    print("\n── Stage 1: Tag Extraction ──")
    tags = run_tag_extraction(df_in)

    # Stage 2
    print("\n── Stage 2: Topic Mapping ──")
    topic_to_tags   = run_topic_mapping(tags, TOPIC_TAXONOMY)
    tt_lookup       = tag_to_topic_lookup(topic_to_tags)
    comment_topics  = derive_comment_topics(tags, tt_lookup)

    # Stage 3
    print("\n── Stage 3: Sentiment Analysis ──")
    sentiment = run_sentiment(df_in, comment_topics)

    # Aggregations
    tag_freq    = tag_frequencies(tags)
    sent_matrix = topic_sentiment_matrix(sentiment)

    return {
        "month":          month,
        "tags":           tags,
        "topic_to_tags":  topic_to_tags,
        "comment_topics": comment_topics,
        "sentiment":      sentiment,
        "tag_freq":       tag_freq,
        "sent_matrix":    sent_matrix,
    }

# ── Load or run current month ──────────────────────────────────
cached_current = load_results(current_month)
if cached_current and ask_reuse(current_month):
    print(f"♻️  Reusing cached results for {current_month}")
    current_result = cached_current
else:
    current_result = run_pipeline_for_month(current_month, df_current)
    save_results(current_month, current_result)

# ── Load or run prior month ────────────────────────────────────
prior_result = None
if prior_month:
    cached_prior = load_results(prior_month)
    if cached_prior:
        if ask_reuse(prior_month):
            print(f"♻️  Reusing cached results for {prior_month}")
            prior_result = cached_prior
        else:
            prior_result = run_pipeline_for_month(prior_month, df_prior)
            save_results(prior_month, prior_result)
    elif not df_prior.empty:
        prior_result = run_pipeline_for_month(prior_month, df_prior)
        save_results(prior_month, prior_result)

HAS_PRIOR = prior_result is not None
print(f"\n✅ Cell 11 — orchestration complete  |  has_prior={HAS_PRIOR}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 12 — Stage 4: Topic Commentary (after data is ready)  ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n── Stage 4: Topic Commentary ──")

current_grouped = group_comments_by_sentiment(df_current, current_result["sentiment"])
prior_grouped   = (
    group_comments_by_sentiment(df_prior, prior_result["sentiment"])
    if HAS_PRIOR and not df_prior.empty
    else {}
)

topic_summaries = run_commentary(current_grouped, prior_grouped, HAS_PRIOR)

# Persist commentary back into result file
current_result["topic_summaries"] = topic_summaries
save_results(current_month, current_result)

print(f"✅ Cell 12 — commentary done for {len(topic_summaries)} topics")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 13 — Build HTML Dashboard                             ║
# ╚══════════════════════════════════════════════════════════════╝

def build_dashboard(
    current_month, prior_month, has_prior,
    current_result, prior_result, topic_summaries
) -> str:

    cur_sent    = current_result["sent_matrix"]
    pri_sent    = prior_result["sent_matrix"] if has_prior else {}
    cur_tags    = dict(list(current_result["tag_freq"].items())[:60])
    pri_tags    = dict(list(prior_result["tag_freq"].items())[:60]) if has_prior else {}
    topics      = sorted(topic_summaries.keys())

    # ── Serialise to JS ────────────────────────────────────────
    topics_js      = json.dumps(topics)
    summaries_js   = json.dumps(topic_summaries)
    sent_cur_js    = json.dumps(cur_sent)
    sent_pri_js    = json.dumps(pri_sent)
    tag_cur_js     = json.dumps(cur_tags)
    tag_pri_js     = json.dumps(pri_tags)
    has_prior_js   = "true" if has_prior else "false"
    cur_label      = current_month
    pri_label      = prior_month or ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comment Intelligence — {cur_label}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#F4F6FA;--surface:#FFFFFF;--surface2:#F0F2F7;
  --border:#E2E6EF;--text:#1A2035;--muted:#6B7A99;
  --accent:#3B6FF0;--accent-light:#EEF2FE;
  --pos:#2DA86A;--pos-light:#E8F8F0;
  --neg:#E04545;--neg-light:#FDEAEA;
  --neu:#8A96B0;--neu-light:#F0F2F7;
  --warn:#F59E0B;--warn-light:#FFFBEB;
  --r:10px;--r-sm:6px;
  --shadow:0 2px 12px rgba(0,0,0,0.07);
  --shadow-md:0 4px 20px rgba(0,0,0,0.10);
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}}

/* ── Header ── */
.header{{background:linear-gradient(135deg,#1A2035 0%,#2C3E6B 100%);padding:28px 40px;display:flex;justify-content:space-between;align-items:center}}
.header-left h1{{font-size:1.4rem;font-weight:700;color:#fff;letter-spacing:-0.3px}}
.header-left p{{font-size:0.78rem;color:rgba(255,255,255,0.55);margin-top:3px;font-family:'IBM Plex Mono',monospace}}
.period-badges{{display:flex;gap:8px}}
.badge{{font-family:'IBM Plex Mono',monospace;font-size:0.68rem;padding:5px 12px;border-radius:20px;font-weight:500;letter-spacing:0.4px}}
.badge-cur{{background:rgba(59,111,240,0.25);color:#93B4FF;border:1px solid rgba(59,111,240,0.4)}}
.badge-pri{{background:rgba(245,158,11,0.2);color:#FCD34D;border:1px solid rgba(245,158,11,0.3)}}

/* ── Layout ── */
.main{{max-width:1200px;margin:0 auto;padding:28px 24px}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.mt{{margin-top:18px}}

/* ── Section headers ── */
.section-header{{display:flex;align-items:center;gap:10px;margin:24px 0 14px}}
.section-header h2{{font-size:0.95rem;font-weight:600;color:var(--text)}}
.section-line{{flex:1;height:1px;background:var(--border)}}

/* ── Cards ── */
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:20px;box-shadow:var(--shadow)}}
.card-title{{font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:14px}}

/* ── Stat cards ── */
.stat-card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:18px 20px;box-shadow:var(--shadow);position:relative;overflow:hidden}}
.stat-card::after{{content:'';position:absolute;bottom:0;left:0;right:0;height:3px}}
.stat-pos::after{{background:var(--pos)}}
.stat-neg::after{{background:var(--neg)}}
.stat-total::after{{background:var(--accent)}}
.stat-topics::after{{background:var(--warn)}}
.stat-label{{font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:var(--muted);font-weight:600}}
.stat-value{{font-size:2rem;font-weight:700;line-height:1;margin:6px 0 4px}}
.stat-sub{{font-size:0.72rem;color:var(--muted)}}
.stat-pos .stat-value{{color:var(--pos)}}
.stat-neg .stat-value{{color:var(--neg)}}
.stat-total .stat-value{{color:var(--accent)}}
.stat-topics .stat-value{{color:var(--warn)}}

/* ── Tabs ── */
.tabs{{display:flex;gap:2px;background:var(--surface2);border-radius:var(--r);padding:4px;margin-bottom:18px;width:fit-content}}
.tab{{padding:8px 18px;border-radius:var(--r-sm);font-size:0.8rem;font-weight:500;color:var(--muted);cursor:pointer;transition:all .2s;user-select:none;white-space:nowrap}}
.tab:hover{{color:var(--text)}}
.tab.active{{background:var(--surface);color:var(--accent);box-shadow:0 1px 4px rgba(0,0,0,0.1);font-weight:600}}
.tab-content{{display:none}}.tab-content.active{{display:block}}

/* ── Sentiment bar rows ── */
.topic-bars{{display:flex;flex-direction:column;gap:10px}}
.tbar-row{{display:grid;grid-template-columns:170px 1fr 110px;align-items:center;gap:12px}}
.tbar-name{{font-size:0.78rem;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}}
.tbar-tracks{{display:flex;flex-direction:column;gap:3px}}
.tbar-track{{background:var(--surface2);border-radius:3px;height:7px;overflow:hidden}}
.tbar-fill{{height:100%;border-radius:3px;transition:width .6s ease}}
.fill-pos{{background:var(--pos)}}.fill-neg{{background:var(--neg)}}.fill-neu{{background:var(--neu)}}
.tbar-meta{{font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:var(--muted);line-height:1.6}}

/* ── Sentiment counts box ── */
.counts-box{{display:flex;gap:8px;justify-content:flex-end}}
.count-chip{{display:flex;flex-direction:column;align-items:center;padding:4px 10px;border-radius:var(--r-sm);font-family:'IBM Plex Mono',monospace}}
.count-chip .cv{{font-size:0.95rem;font-weight:600;line-height:1}}
.count-chip .cp{{font-size:0.6rem;margin-top:1px}}
.chip-pos{{background:var(--pos-light);color:var(--pos)}}
.chip-neg{{background:var(--neg-light);color:var(--neg)}}
.chip-neu{{background:var(--neu-light);color:var(--neu)}}

/* ── Tag heatmap ── */
.tag-cloud{{display:flex;flex-wrap:wrap;gap:7px;align-items:center}}
.tag-chip{{display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:var(--r-sm);font-size:0.72rem;font-family:'IBM Plex Mono',monospace;cursor:default;transition:transform .12s}}
.tag-chip:hover{{transform:translateY(-1px)}}
.tc-count{{opacity:.7;font-size:0.62rem}}

/* ── Word cloud ── */
.wordcloud{{position:relative;width:100%;height:220px;overflow:hidden}}
.wc-word{{position:absolute;transform-origin:center;cursor:default;font-weight:600;transition:opacity .2s;white-space:nowrap;font-family:'Inter',sans-serif}}
.wc-word:hover{{opacity:0.7}}

/* ── Pie donut chart (SVG) ── */
.donut-wrap{{display:flex;align-items:center;justify-content:center;gap:24px}}
.donut-legend{{display:flex;flex-direction:column;gap:8px}}
.dl-item{{display:flex;align-items:center;gap:8px;font-size:0.76rem}}
.dl-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.dl-val{{font-family:'IBM Plex Mono',monospace;font-weight:600;min-width:30px}}

/* ── Bubble chart (SVG) ── */
.bubble-wrap{{width:100%;height:260px;position:relative;overflow:hidden}}
svg.bubble-svg{{width:100%;height:100%}}

/* ── Summary cards ── */
.summary-cards{{display:flex;flex-direction:column;gap:12px}}
.scard{{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;box-shadow:var(--shadow)}}
.scard-header{{padding:14px 18px;display:flex;align-items:center;justify-content:space-between;cursor:pointer;user-select:none;transition:background .15s}}
.scard-header:hover{{background:var(--surface2)}}
.scard-title{{font-size:0.9rem;font-weight:600;color:var(--text)}}
.scard-meta{{display:flex;align-items:center;gap:8px}}
.pill{{font-size:0.62rem;font-weight:600;padding:3px 9px;border-radius:10px;font-family:'IBM Plex Mono',monospace}}
.pill-pos{{background:var(--pos-light);color:var(--pos)}}
.pill-neg{{background:var(--neg-light);color:var(--neg)}}
.pill-neu{{background:var(--neu-light);color:var(--neu)}}
.pill-status-ok{{background:#E8F8F0;color:#2DA86A;border:1px solid #B6E8CE}}
.pill-status-warn{{background:#FDEAEA;color:#E04545;border:1px solid #F5BBBB}}
.pill-status-mid{{background:#FFFBEB;color:#B45309;border:1px solid #FCD34D}}
.chevron{{color:var(--muted);font-size:0.75rem;transition:transform .2s}}
.scard-body{{max-height:0;overflow:hidden;transition:max-height .35s ease,padding .2s}}
.scard-body.open{{max-height:700px;padding:0 18px 18px}}
.sent-block{{margin-top:14px}}
.sent-block-label{{font-size:0.65rem;text-transform:uppercase;letter-spacing:1px;font-weight:700;display:flex;align-items:center;gap:6px;margin-bottom:5px}}
.sent-dot{{width:6px;height:6px;border-radius:50%;flex-shrink:0}}
.sent-text{{font-size:0.82rem;line-height:1.65;color:#4A5568}}
.commentary-box{{background:var(--accent-light);border:1px solid #C7D7FB;border-radius:8px;padding:12px 14px;margin-top:12px}}
.commentary-label{{font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;color:var(--accent);font-weight:700;margin-bottom:4px}}
.commentary-text{{font-size:0.82rem;line-height:1.65;color:#3B5299}}
.mom-box{{background:var(--warn-light);border:1px solid #FDE68A;border-radius:8px;padding:12px 14px;margin-top:10px}}
.mom-label{{font-size:0.62rem;text-transform:uppercase;letter-spacing:1px;color:var(--warn);font-weight:700;margin-bottom:4px}}
.mom-text{{font-size:0.82rem;line-height:1.65;color:#92400E}}

/* ── Filter chips ── */
.filter-row{{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px}}
.fchip{{padding:4px 12px;border-radius:20px;font-size:0.72rem;font-weight:500;border:1px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .15s}}
.fchip:hover,.fchip.active{{border-color:var(--accent);color:var(--accent);background:var(--accent-light)}}

/* ── Legend row ── */
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}}
.leg-item{{display:flex;align-items:center;gap:5px;font-size:0.72rem;color:var(--muted)}}
.leg-dot{{width:8px;height:8px;border-radius:50%;flex-shrink:0}}

/* ── Scroll ── */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--surface2)}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <h1>Comment Intelligence Dashboard</h1>
    <p>AI-extracted topics · tags · sentiment analysis</p>
  </div>
  <div class="period-badges">
    <span class="badge badge-cur">Current: {cur_label}</span>
    {'<span class="badge badge-pri">Prior: ' + pri_label + '</span>' if pri_label else ''}
  </div>
</div>

<div class="main">
  <!-- STAT CARDS -->
  <div class="grid-4" id="stat-cards"></div>

  <!-- TABS -->
  <div class="mt">
    <div class="tabs">
      <div class="tab active" onclick="switchTab(0)">📊 Sentiment Trends</div>
      <div class="tab" onclick="switchTab(1)">🏷️ Tag Analysis</div>
      <div class="tab" onclick="switchTab(2)">📋 Topic Summaries</div>
    </div>

    <!-- TAB 0: Sentiment Trends -->
    <div class="tab-content active" id="tab-0">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Sentiment Distribution — {cur_label}</div>
          <div class="donut-wrap" id="donut-cur"></div>
        </div>
        {'<div class="card"><div class="card-title">Sentiment Distribution — ' + pri_label + '</div><div class="donut-wrap" id="donut-pri"></div></div>' if has_prior else '<div></div>'}
      </div>

      <div class="card mt">
        <div class="card-title">Topic-Level Sentiment Breakdown</div>
        <div class="legend">
          <div class="leg-item"><div class="leg-dot" style="background:var(--pos)"></div>Positive</div>
          <div class="leg-item"><div class="leg-dot" style="background:var(--neg)"></div>Negative</div>
          <div class="leg-item"><div class="leg-dot" style="background:var(--neu)"></div>Neutral</div>
          {'<div class="leg-item" style="margin-left:auto;font-size:.68rem;color:var(--muted)">Outlined = prior month</div>' if has_prior else ''}
        </div>
        <div class="filter-row" id="topic-filter"></div>
        <div class="topic-bars" id="topic-bars"></div>
      </div>

      <div class="card mt">
        <div class="card-title">% Negative vs % Positive — Bubble View (size = total comments)</div>
        <div class="bubble-wrap"><svg class="bubble-svg" id="bubble-svg"></svg></div>
      </div>
    </div>

    <!-- TAB 1: Tag Analysis -->
    <div class="tab-content" id="tab-1">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Tag Frequency Heatmap — {cur_label}</div>
          <div class="tag-cloud" id="heatmap-cur"></div>
        </div>
        {'<div class="card"><div class="card-title">Tag Frequency Heatmap — ' + pri_label + '</div><div class="tag-cloud" id="heatmap-pri"></div></div>' if has_prior else '<div class="card"><div class="card-title">Word Cloud — ' + cur_label + '</div><div class="wordcloud" id="wordcloud-solo"></div></div>'}
      </div>
      <div class="grid-2 mt">
        <div class="card">
          <div class="card-title">Word Cloud — {cur_label}</div>
          <div class="wordcloud" id="wordcloud-cur"></div>
        </div>
        {'<div class="card"><div class="card-title">Word Cloud — ' + pri_label + '</div><div class="wordcloud" id="wordcloud-pri"></div></div>' if has_prior else '<div></div>'}
      </div>
    </div>

    <!-- TAB 2: Topic Summaries -->
    <div class="tab-content" id="tab-2">
      <div class="summary-cards" id="summary-cards"></div>
    </div>
  </div>
</div>

<script>
// ── Data ─────────────────────────────────────────────────────
const TOPICS       = {topics_js};
const SUMMARIES    = {summaries_js};
const SENT_CUR     = {sent_cur_js};
const SENT_PRI     = {sent_pri_js};
const TAG_CUR      = {tag_cur_js};
const TAG_PRI      = {tag_pri_js};
const HAS_PRIOR    = {has_prior_js};
const CUR_LABEL    = "{cur_label}";
const PRI_LABEL    = "{pri_label}";
const C_POS='#2DA86A', C_NEG='#E04545', C_NEU='#8A96B0';
const C_POS_L='#E8F8F0', C_NEG_L='#FDEAEA', C_NEU_L='#F0F2F7';

// ── Utils ─────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const ce = (tag, cls='', html='') => {{
  const el = document.createElement(tag);
  if(cls)  el.className = cls;
  if(html) el.innerHTML = html;
  return el;
}};

function sentTotals(matrix){{
  let p=0,n=0,u=0;
  Object.values(matrix).forEach(t=>{{ p+=t.positive||0; n+=t.negative||0; u+=t.neutral||0; }});
  return {{p,n,u}};
}}
function pct(a,b){{ return b?Math.round(a/b*100):0; }}
function statusPill(pctNeg){{
  if(pctNeg>=50) return ['pill-status-warn','High Concern'];
  if(pctNeg>=25) return ['pill-status-mid','Moderate'];
  return ['pill-status-ok','Healthy'];
}}

// ── Tabs ─────────────────────────────────────────────────────
function switchTab(n){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===n));
  document.querySelectorAll('.tab-content').forEach((c,i)=>c.classList.toggle('active',i===n));
}}

// ── Stat cards ────────────────────────────────────────────────
function buildStats(){{
  const t = sentTotals(SENT_CUR);
  const total = t.p+t.n+t.u;
  const cards = [
    {{cls:'stat-total',label:'Total Comments',val:total,sub:CUR_LABEL}},
    {{cls:'stat-pos',label:'Positive',val:t.p,sub:`${{pct(t.p,total)}}% of comments`}},
    {{cls:'stat-neg',label:'Negative',val:t.n,sub:`${{pct(t.n,total)}}% of comments`}},
    {{cls:'stat-topics',label:'Topics Found',val:TOPICS.length,sub:'across current month'}},
  ];
  $('stat-cards').innerHTML = cards.map(c=>`
    <div class="stat-card ${{c.cls}}">
      <div class="stat-label">${{c.label}}</div>
      <div class="stat-value">${{c.val}}</div>
      <div class="stat-sub">${{c.sub}}</div>
    </div>`).join('');
}}

// ── SVG donut ─────────────────────────────────────────────────
function buildDonut(containerId, matrix, label){{
  const t = sentTotals(matrix);
  const total = t.p+t.n+t.u;
  if(!total) return;
  const slices = [
    {{val:t.p,color:C_POS,label:'Positive'}},
    {{val:t.n,color:C_NEG,label:'Negative'}},
    {{val:t.u,color:C_NEU,label:'Neutral'}},
  ];
  const R=70, cx=90, cy=90, r=48, inner=30;
  let angle = -Math.PI/2;
  let paths = '';
  slices.forEach(s=>{{
    if(!s.val) return;
    const a = (s.val/total)*2*Math.PI;
    const x1=cx+R*Math.cos(angle)*r/R, y1=cy+R*Math.sin(angle)*r/R;
    angle+=a;
    const x2=cx+R*Math.cos(angle)*r/R, y2=cy+R*Math.sin(angle)*r/R;
    const lg = a>Math.PI?1:0;
    // outer arc
    const ox1=cx+r*Math.cos(angle-a), oy1=cy+r*Math.sin(angle-a);
    const ox2=cx+r*Math.cos(angle),   oy2=cy+r*Math.sin(angle);
    const ix1=cx+inner*Math.cos(angle), iy1=cy+inner*Math.sin(angle);
    const ix2=cx+inner*Math.cos(angle-a), iy2=cy+inner*Math.sin(angle-a);
    paths+=`<path d="M${{ox1}},${{oy1}} A${{r}},${{r}} 0 ${{lg}},1 ${{ox2}},${{oy2}} L${{ix1}},${{iy1}} A${{inner}},${{inner}} 0 ${{lg}},0 ${{ix2}},${{iy2}} Z"
      fill="${{s.color}}" opacity="0.9"/>`;
  }});
  const legendHtml = slices.map(s=>`
    <div class="dl-item">
      <div class="dl-dot" style="background:${{s.color}}"></div>
      <span class="dl-val">${{s.val}}</span>
      <span style="color:var(--muted)">${{s.label}} (${{pct(s.val,total)}}%)</span>
    </div>`).join('');
  const el = $(containerId);
  if(!el) return;
  el.innerHTML = `
    <svg width="180" height="180" viewBox="0 0 180 180">
      ${{paths}}
      <text x="90" y="86" text-anchor="middle" font-size="22" font-weight="700" fill="#1A2035">${{total}}</text>
      <text x="90" y="100" text-anchor="middle" font-size="9" fill="#6B7A99">COMMENTS</text>
    </svg>
    <div class="donut-legend">${{legendHtml}}</div>`;
}}

// ── Topic bars ────────────────────────────────────────────────
let activeFilter = null;
function buildFilterBtns(){{
  const el = $('topic-filter');
  const all = ce('button','fchip active','All');
  all.onclick = () => {{ activeFilter=null; document.querySelectorAll('.fchip').forEach(b=>b.classList.remove('active')); all.classList.add('active'); renderBars(); }};
  el.appendChild(all);
  TOPICS.filter(t=>SENT_CUR[t]).forEach(t=>{{
    const btn = ce('button','fchip',t);
    btn.onclick = () => {{ activeFilter=t; document.querySelectorAll('.fchip').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); renderBars(); }};
    el.appendChild(btn);
  }});
}}

function renderBars(){{
  const wrap = $('topic-bars');
  const list = activeFilter ? [activeFilter] : TOPICS.filter(t=>SENT_CUR[t]);
  wrap.innerHTML = list.map(topic=>{{
    const c = SENT_CUR[topic]||{{}};
    const p = SENT_PRI[topic]||{{}};
    const cp=c.positive||0, cn=c.negative||0, cne=c.neutral||0;
    const total = cp+cn+cne; if(!total) return '';
    const pctP=pct(cp,total), pctN=pct(cn,total), pctNe=pct(cne,total);
    const priLine = HAS_PRIOR && (p.positive||p.negative)
      ? `<div style="font-size:.63rem;color:var(--muted);margin-top:1px">Prior: +${{p.positive||0}} (${{pct(p.positive||0,(p.positive||0)+(p.negative||0)+(p.neutral||0))||0}}%) / -${{p.negative||0}} (${{pct(p.negative||0,(p.positive||0)+(p.negative||0)+(p.neutral||0))||0}}%)</div>`
      : '';
    return `<div class="tbar-row">
      <div class="tbar-name" title="${{topic}}">${{topic}}</div>
      <div>
        <div class="tbar-tracks">
          <div class="tbar-track"><div class="tbar-fill fill-pos" style="width:${{pctP}}%"></div></div>
          <div class="tbar-track"><div class="tbar-fill fill-neg" style="width:${{pctN}}%"></div></div>
          <div class="tbar-track"><div class="tbar-fill fill-neu" style="width:${{pctNe}}%"></div></div>
        </div>
        ${{priLine}}
      </div>
      <div class="counts-box">
        <div class="count-chip chip-pos"><span class="cv">${{cp}}</span><span class="cp">${{pctP}}% pos</span></div>
        <div class="count-chip chip-neg"><span class="cv">${{cn}}</span><span class="cp">${{pctN}}% neg</span></div>
      </div>
    </div>`;
  }}).join('');
}}

// ── Bubble chart (pure SVG) ───────────────────────────────────
function buildBubble(){{
  const svg = $('bubble-svg');
  if(!svg) return;
  const W=svg.clientWidth||800, H=260;
  const topics = TOPICS.filter(t=>SENT_CUR[t]);
  const rows = topics.map(t=>{{
    const c=SENT_CUR[t]||{{}};
    const total=(c.positive||0)+(c.negative||0)+(c.neutral||0);
    return {{topic:t, total, pctP:pct(c.positive||0,total), pctN:pct(c.negative||0,total)}};
  }}).filter(r=>r.total>0);
  const maxT = Math.max(...rows.map(r=>r.total));
  const PAD=40;
  // Axes
  let html=`
    <line x1="${{PAD}}" y1="${{H-PAD}}" x2="${{W-PAD}}" y2="${{H-PAD}}" stroke="#E2E6EF" stroke-width="1"/>
    <line x1="${{PAD}}" y1="${{PAD}}" x2="${{PAD}}" y2="${{H-PAD}}" stroke="#E2E6EF" stroke-width="1"/>
    <text x="${{W/2}}" y="${{H-8}}" text-anchor="middle" font-size="10" fill="#6B7A99">% Positive</text>
    <text x="12" y="${{H/2}}" text-anchor="middle" font-size="10" fill="#6B7A99" transform="rotate(-90,12,${{H/2}})">% Negative</text>`;
  // Grid lines
  [25,50,75].forEach(v=>{{
    const gx=PAD+(v/100)*(W-2*PAD);
    const gy=H-PAD-(v/100)*(H-2*PAD);
    html+=`<line x1="${{gx}}" y1="${{PAD}}" x2="${{gx}}" y2="${{H-PAD}}" stroke="#E2E6EF" stroke-width="1" stroke-dasharray="3,3"/>`;
    html+=`<line x1="${{PAD}}" y1="${{gy}}" x2="${{W-PAD}}" y2="${{gy}}" stroke="#E2E6EF" stroke-width="1" stroke-dasharray="3,3"/>`;
    html+=`<text x="${{gx}}" y="${{H-PAD+12}}" text-anchor="middle" font-size="8" fill="#9AA3B8">${{v}}%</text>`;
    html+=`<text x="${{PAD-6}}" y="${{gy+3}}" text-anchor="end" font-size="8" fill="#9AA3B8">${{v}}%</text>`;
  }});
  const COLORS=['#3B6FF0','#E04545','#2DA86A','#F59E0B','#8B5CF6','#EC4899','#06B6D4','#84CC16','#F97316','#6366F1'];
  rows.forEach((r,i)=>{{
    const x=PAD+(r.pctP/100)*(W-2*PAD);
    const y=H-PAD-(r.pctN/100)*(H-2*PAD);
    const rad=Math.max(8,Math.min(28,(r.total/maxT)*30));
    const col=COLORS[i%COLORS.length];
    const short=r.topic.split(' ').slice(0,2).join(' ');
    html+=`<circle cx="${{x}}" cy="${{y}}" r="${{rad}}" fill="${{col}}" opacity="0.75"/>
      <text x="${{x}}" y="${{y+3}}" text-anchor="middle" font-size="8" fill="white" font-weight="600">${{short}}</text>`;
  }});
  svg.innerHTML=html;
}}

// ── Tag heatmap ───────────────────────────────────────────────
const TC_PALETTES=[
  {{bg:'rgba(59,111,240,0.15)',color:'#2850C0',border:'rgba(59,111,240,0.3)'}},
  {{bg:'rgba(45,168,106,0.15)',color:'#1A7A4A',border:'rgba(45,168,106,0.3)'}},
  {{bg:'rgba(245,158,11,0.15)',color:'#92400E',border:'rgba(245,158,11,0.3)'}},
  {{bg:'rgba(139,92,246,0.15)',color:'#5B21B6',border:'rgba(139,92,246,0.3)'}},
  {{bg:'rgba(236,72,153,0.15)',color:'#9D174D',border:'rgba(236,72,153,0.3)'}},
];

function buildHeatmap(containerId, tagFreq){{
  const el=$(containerId); if(!el) return;
  const entries=Object.entries(tagFreq);
  const maxF=entries[0]?.[1]||1;
  el.innerHTML=entries.map(([tag,freq],i)=>{{
    const ratio=freq/maxF;
    const size=ratio>0.7?'1rem':ratio>0.4?'0.82rem':'0.72rem';
    const pal=TC_PALETTES[i%TC_PALETTES.length];
    return `<div class="tag-chip" style="background:${{pal.bg}};color:${{pal.color}};border:1px solid ${{pal.border}};font-size:${{size}}">
      ${{tag}}<span class="tc-count">(${{freq}})</span>
    </div>`;
  }}).join('');
}}

// ── Word cloud (pure CSS/JS positioned) ──────────────────────
function buildWordCloud(containerId, tagFreq){{
  const el=$(containerId); if(!el) return;
  const entries=Object.entries(tagFreq).slice(0,40);
  if(!entries.length) return;
  const maxF=entries[0][1];
  const W=el.offsetWidth||400, H=220;
  const WC_COLORS=['#3B6FF0','#E04545','#2DA86A','#F59E0B','#8B5CF6','#EC4899','#06B6D4','#F97316'];
  const placed=[];
  function overlaps(x,y,w,h){{
    for(const p of placed){{
      if(x<p.x+p.w && x+w>p.x && y<p.y+p.h && y+h>p.y) return true;
    }}
    return false;
  }}
  let html='';
  entries.forEach(([tag,freq],i)=>{{
    const ratio=freq/maxF;
    const fs=Math.round(10+ratio*22);
    const color=WC_COLORS[i%WC_COLORS.length];
    const estW=tag.length*fs*0.55, estH=fs*1.4;
    let placed_=false;
    for(let attempt=0;attempt<120;attempt++){{
      const angle=attempt*2.4;
      const r=attempt*3.5;
      const cx=W/2+r*Math.cos(angle)-estW/2;
      const cy=H/2+r*Math.sin(angle)*0.6-estH/2;
      if(cx<2||cy<2||cx+estW>W-2||cy+estH>H-2) continue;
      if(!overlaps(cx,cy,estW,estH)){{
        placed.push({{x:cx,y:cy,w:estW,h:estH}});
        html+=`<span class="wc-word" style="left:${{cx}}px;top:${{cy}}px;font-size:${{fs}}px;color:${{color}};opacity:${{0.55+ratio*0.45}}">${{tag}}</span>`;
        placed_=true; break;
      }}
    }}
  }});
  el.innerHTML=html;
}}

// ── Summary cards ─────────────────────────────────────────────
function buildSummaryCards(){{
  const wrap=$('summary-cards');
  wrap.innerHTML=TOPICS.map((topic,i)=>{{
    const d=SUMMARIES[topic]; if(!d) return '';
    const s=d.summary;
    const cc=d.current_counts||{{}};
    const total=(cc.positive||0)+(cc.negative||0)+(cc.neutral||0)||1;
    const pctN=pct(cc.negative||0,total);
    const [statusCls,statusLabel]=statusPill(pctN);

    const pills=[
      cc.positive?`<span class="pill pill-pos">+${{cc.positive}} (${{pct(cc.positive||0,total)}}%)</span>`:'',
      cc.negative?`<span class="pill pill-neg">-${{cc.negative}} (${{pctN}}%)</span>`:'',
      cc.neutral?`<span class="pill pill-neu">≈${{cc.neutral}}</span>`:'',
    ].filter(Boolean).join('');

    const blocks=[];
    if(s.positive_summary) blocks.push(`
      <div class="sent-block">
        <div class="sent-block-label"><div class="sent-dot" style="background:${{C_POS}}"></div><span style="color:${{C_POS}}">Positive Observations</span></div>
        <div class="sent-text">${{s.positive_summary}}</div>
      </div>`);
    if(s.negative_summary) blocks.push(`
      <div class="sent-block">
        <div class="sent-block-label"><div class="sent-dot" style="background:${{C_NEG}}"></div><span style="color:${{C_NEG}}">Negative Observations</span></div>
        <div class="sent-text">${{s.negative_summary}}</div>
      </div>`);
    if(s.neutral_summary) blocks.push(`
      <div class="sent-block">
        <div class="sent-block-label"><div class="sent-dot" style="background:${{C_NEU}}"></div><span style="color:${{C_NEU}}">Neutral Observations</span></div>
        <div class="sent-text">${{s.neutral_summary}}</div>
      </div>`);
    if(s.commentary) blocks.push(`
      <div class="commentary-box">
        <div class="commentary-label">💡 Topic Commentary</div>
        <div class="commentary-text">${{s.commentary}}</div>
      </div>`);
    if(s.mom_commentary) blocks.push(`
      <div class="mom-box">
        <div class="mom-label">📈 Month-over-Month</div>
        <div class="mom-text">${{s.mom_commentary}}</div>
      </div>`);

    return `<div class="scard">
      <div class="scard-header" onclick="toggleCard(${{i}})">
        <div class="scard-title">${{topic}}</div>
        <div class="scard-meta">
          ${{pills}}
          <span class="pill ${{statusCls}}">${{statusLabel}}</span>
          <span class="chevron" id="chev-${{i}}">▼</span>
        </div>
      </div>
      <div class="scard-body" id="sbody-${{i}}">${{blocks.join('')}}</div>
    </div>`;
  }}).join('');
}}

function toggleCard(i){{
  const b=$('sbody-'+i), c=$('chev-'+i);
  const open=b.classList.toggle('open');
  if(c) c.style.transform=open?'rotate(180deg)':'';
}}

// ── Boot ──────────────────────────────────────────────────────
buildStats();
buildDonut('donut-cur', SENT_CUR, CUR_LABEL);
if(HAS_PRIOR) buildDonut('donut-pri', SENT_PRI, PRI_LABEL);
buildFilterBtns();
renderBars();
buildHeatmap('heatmap-cur', TAG_CUR);
if(HAS_PRIOR) buildHeatmap('heatmap-pri', TAG_PRI);
buildSummaryCards();

// Word clouds need layout first
setTimeout(()=>{{
  buildBubble();
  buildWordCloud('wordcloud-cur', TAG_CUR);
  if(HAS_PRIOR) buildWordCloud('wordcloud-pri', TAG_PRI);
  else buildWordCloud('wordcloud-solo', TAG_CUR);
  // Auto-open first summary
  const b=$('sbody-0'); if(b){{ b.classList.add('open'); const c=$('chev-0'); if(c) c.style.transform='rotate(180deg)'; }}
}},300);
</script>
</body>
</html>"""

print("✅ Cell 13 — dashboard builder defined")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 14 — Render & Export Dashboard                        ║
# ╚══════════════════════════════════════════════════════════════╝

html_out = build_dashboard(
    current_month  = current_month,
    prior_month    = prior_month,
    has_prior      = HAS_PRIOR,
    current_result = current_result,
    prior_result   = prior_result,
    topic_summaries= topic_summaries,
)

# ── Display inline in notebook ─────────────────────────────────
display(HTML(html_out))

# ── Save HTML file ─────────────────────────────────────────────
html_path = os.path.join(RESULTS_ROOT, f"dashboard_{current_month}.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_out)

print(f"✅ Cell 14 — dashboard rendered & saved to: {html_path}")
