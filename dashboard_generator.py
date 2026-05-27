"""
NPS Dashboard Generator
Takes a DataFrame with Survey Month, Comments, Topics, Subtopics, Sentiment columns.
LLM calls are only for narrative summaries (sentiment is already in the data).
Handles deduplication: same comment with multiple subtopics = one unique comment.
"""

import os
import re
import json
import time
import random
import pandas as pd
from google import genai
from collections import defaultdict, Counter

# ============================================================
# CONFIGURATION - Set these values
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"
MODEL = "gemma-4-26b-a4b-it"
DATA_FILE = "data.xlsx"
SHEET_NAME = "Sheet1"

# Column names in your DataFrame
MONTH_COL = "Survey Month"
COMMENT_COL = "Comments"
TOPIC_COL = "Topics"
SUBTOPIC_COL = "Subtopics"
SENTIMENT_COL = "Sentiment"  # Values: POSITIVE, NEGATIVE, NEUTRAL

# Output
OUTPUT_HTML = "dashboard_output.html"
TEMPLATE_HTML = os.path.join(os.path.dirname(__file__), "dashboard_template_simple.html")

# ============================================================
# SETUP
# ============================================================
client = genai.Client(api_key=API_KEY)

def call_llm(system_prompt, user_prompt, label="call"):
    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config={"system_instruction": system_prompt, "temperature": 0.0}
    )
    return response.text.strip()

def parse_json(text, label="response"):
    cleaned = text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned)
    cleaned = re.sub(r"```\s*", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse {label}: {e}")
        return None

def dedup_comments(comment_list):
    """Deduplicate by comment text. Returns list of unique comments."""
    seen = set()
    unique = []
    for c in comment_list:
        key = c["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# ============================================================
# CELL 1: LOAD DATA
# ============================================================
print("=" * 60)
print("CELL 1: LOAD DATA")
print("=" * 60)

if DATA_FILE.endswith(".xlsx") or DATA_FILE.endswith(".xls"):
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
else:
    df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df)} rows from {DATA_FILE}")
print(f"Columns: {list(df.columns)}")

# Validate columns
for col in [MONTH_COL, COMMENT_COL, TOPIC_COL, SUBTOPIC_COL, SENTIMENT_COL]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

# Detect current and prior months
months = sorted(df[MONTH_COL].unique())
CURRENT_MONTH = months[-1]
PRIOR_MONTH = months[-2] if len(months) > 1 else None
print(f"Current month: {CURRENT_MONTH}")
print(f"Prior month: {PRIOR_MONTH}")

# Build row-level data
rows = []
for _, row in df.iterrows():
    rows.append({
        "month": str(row[MONTH_COL]).strip(),
        "text": str(row[COMMENT_COL]).strip(),
        "topic": str(row[TOPIC_COL]).strip(),
        "subtopic": str(row[SUBTOPIC_COL]).strip() if pd.notna(row[SUBTOPIC_COL]) else "",
        "sentiment": str(row[SENTIMENT_COL]).strip().upper()
    })

print(f"Unique comments: {len(set(r['text'].lower() for r in rows))} (across all topics)")
print(f"Sentiment distribution: {dict(Counter(r['sentiment'] for r in dedup_comments(rows)))}")

# ============================================================
# CELL 2: GROUP BY TOPIC & COMPUTE SENTIMENT FROM DATA
# ============================================================
print("\n" + "=" * 60)
print("CELL 2: GROUP BY TOPIC & COMPUTE SENTIMENT")
print("=" * 60)

# Group rows by topic
topic_rows = defaultdict(list)
for r in rows:
    topic_rows[r["topic"]].append(r)

# For each topic, deduplicate and count
topic_stats = {}
for topic, topic_rs in sorted(topic_rows.items()):
    unique_topic = dedup_comments(topic_rs)
    subtopics = sorted(set(r["subtopic"] for r in topic_rs if r["subtopic"]))
    sent_counts = Counter(r["sentiment"] for r in unique_topic)

    topic_stats[topic] = {
        "rows": unique_topic,
        "total": len(unique_topic),
        "positive": sent_counts.get("POSITIVE", 0),
        "negative": sent_counts.get("NEGATIVE", 0),
        "neutral": sent_counts.get("NEUTRAL", 0),
        "subtopics": subtopics
    }
    print(f"  {topic}: {topic_stats[topic]['total']} uniq | "
          f"+{topic_stats[topic]['positive']} ~{topic_stats[topic]['neutral']} -{topic_stats[topic]['negative']}")

# Overall unique comments across all topics
all_unique = dedup_comments(rows)
overall_sent = Counter(r["sentiment"] for r in all_unique)
print(f"\nOverall: {len(all_unique)} unique comments | "
      f"+{overall_sent.get('POSITIVE',0)} ~{overall_sent.get('NEUTRAL',0)} -{overall_sent.get('NEGATIVE',0)}")

# ============================================================
# CELL 3: PICK REPRESENTATIVE QUOTES FROM DATA
# ============================================================
print("\n" + "=" * 60)
print("CELL 3: PICK REPRESENTATIVE QUOTES")
print("=" * 60)

topic_quotes = {}
for topic, st in topic_stats.items():
    pos_quotes = [r["text"] for r in st["rows"] if r["sentiment"] == "POSITIVE"]
    neg_quotes = [r["text"] for r in st["rows"] if r["sentiment"] == "NEGATIVE"]
    random.shuffle(pos_quotes)
    random.shuffle(neg_quotes)
    topic_quotes[topic] = {
        "positive": pos_quotes[:2],
        "negative": neg_quotes[:2]
    }
    print(f"  {topic}: {len(topic_quotes[topic]['positive'])} pos, {len(topic_quotes[topic]['negative'])} neg quotes")

# ============================================================
# CELL 4: LLM - GENERATE TOPIC NARRATIVES
# ============================================================
print("\n" + "=" * 60)
print("CELL 4: LLM - GENERATE TOPIC NARRATIVES")
print("=" * 60)

SYS_NARRATIVE = """You are a health insurance member experience analyst.
Write concise, evidence-driven executive summaries of NPS feedback.
Be specific about patterns, cite sentiment data, and highlight actionable insights."""

topic_narratives = {}

for topic, st in sorted(topic_stats.items()):
    print(f"\n  Generating narrative for: {topic}")

    # Build comment block for LLM (deduped)
    comment_block = "\n".join(
        f"  [{i+1}] ({r['sentiment']}, {r['month']}, subtopic: {r['subtopic']}) {r['text'][:300]}"
        for i, r in enumerate(st["rows"])
    )

    prompt = f"""Topic: "{topic}"
Total unique comments: {st["total"]}
Sentiment breakdown: +{st["positive"]} / ~{st["neutral"]} / -{st["negative"]}
Subtopics: {st["subtopics"]}

Comments:
{comment_block}

Write a 4-6 sentence executive summary covering:
- Key patterns in the feedback
- What is driving positive and negative sentiment
- Actionable insights

Return JSON: {{"narrative": "your 4-6 sentence summary here"}}"""

    raw = call_llm(SYS_NARRATIVE, prompt, f"Narrative: {topic}")
    result = parse_json(raw, f"Narrative: {topic}")

    topic_narratives[topic] = {
        "topic": topic,
        "total_comments": st["total"],
        "sentiment_split": {
            "positive": st["positive"],
            "negative": st["negative"],
            "neutral": st["neutral"]
        },
        "subtopics": st["subtopics"],
        "narrative": result.get("narrative", "") if result else "",
        "representative_quotes": topic_quotes[topic]
    }
    print(f"  Narrative length: {len(topic_narratives[topic]['narrative'])} chars")

    time.sleep(2)

print(f"\nGenerated {len(topic_narratives)} topic narratives")

# ============================================================
# CELL 5: LLM - GENERATE MOM COMPARISONS
# ============================================================
print("\n" + "=" * 60)
print("CELL 5: LLM - GENERATE MONTH-OVER-MONTH COMPARISONS")
print("=" * 60)

SYS_MOM = """You are a health insurance member experience analyst.
Compare NPS sentiment between two months and identify key signals and shifts.
Be concise, data-driven, and specific."""

mom_comparisons = {}

if PRIOR_MONTH:
    for topic in sorted(topic_stats.keys()):
        # Get rows for this topic, split by month, deduplicate per month
        topic_rs = topic_rows[topic]
        curr_rs = dedup_comments([r for r in topic_rs if r["month"] == CURRENT_MONTH])
        prior_rs = dedup_comments([r for r in topic_rs if r["month"] == PRIOR_MONTH])

        if not curr_rs or not prior_rs:
            print(f"  Skipping {topic}: no data in both months")
            continue

        curr_sent = Counter(r["sentiment"] for r in curr_rs)
        prior_sent = Counter(r["sentiment"] for r in prior_rs)

        print(f"\n  Comparing {topic}: Current={len(curr_rs)} | Prior={len(prior_rs)}")

        # Build compact prompt with sentiment data + sample comments
        curr_sample = "\n".join(f"  [{r['sentiment']}] {r['text'][:250]}" for r in curr_rs[:8])
        prior_sample = "\n".join(f"  [{r['sentiment']}] {r['text'][:250]}" for r in prior_rs[:8])

        prompt = f"""Topic: "{topic}"

  CURRENT MONTH ({CURRENT_MONTH}) - {len(curr_rs)} unique comments
  Sentiment: +{curr_sent.get("POSITIVE",0)} / ~{curr_sent.get("NEUTRAL",0)} / -{curr_sent.get("NEGATIVE",0)}
  Sample comments:
{curr_sample}

  PRIOR MONTH ({PRIOR_MONTH}) - {len(prior_rs)} unique comments
  Sentiment: +{prior_sent.get("POSITIVE",0)} / ~{prior_sent.get("NEUTRAL",0)} / -{prior_sent.get("NEGATIVE",0)}
  Sample comments:
{prior_sample}

Based on the sentiment data and comments:
1. Determine direction: IMPROVED (negative decreased / positive increased), WORSENED (negative increased / positive decreased), or STABLE (no meaningful change)
2. Write a 2-3 sentence commentary explaining the shift
3. Write a 1-2 sentence key signal for leadership

Return JSON:
{{
  "direction": "IMPROVED or WORSENED or STABLE",
  "commentary": "2-3 sentence shift explanation",
  "key_signals": "1-2 sentence key takeaway"
}}"""

        raw = call_llm(SYS_MOM, prompt, f"MoM: {topic}")
        result = parse_json(raw, f"MoM: {topic}")

        # Also generate prior month narrative summary
        prior_block = "\n".join(f"  [{r['sentiment']}] {r['text'][:250]}" for r in prior_rs[:10])
        prior_prompt = f"""Topic: "{topic}"
Summarize this prior month feedback in 3-4 sentences.

Comments ({PRIOR_MONTH}):
{prior_block}

Return JSON: {{"narrative": "3-4 sentence summary"}}"""

        prior_raw = call_llm(SYS_NARRATIVE, prior_prompt, f"Prior: {topic}")
        prior_result = parse_json(prior_raw, f"Prior: {topic}")

        mom_comparisons[topic] = {
            "topic": topic,
            "prior_narrative": prior_result.get("narrative", "") if prior_result else "",
            "sentiment_shift": {
                "prior": {
                    "positive": prior_sent.get("POSITIVE", 0),
                    "negative": prior_sent.get("NEGATIVE", 0),
                    "neutral": prior_sent.get("NEUTRAL", 0)
                },
                "current": {
                    "positive": curr_sent.get("POSITIVE", 0),
                    "negative": curr_sent.get("NEGATIVE", 0),
                    "neutral": curr_sent.get("NEUTRAL", 0)
                },
                "direction": result.get("direction", "STABLE") if result else "STABLE",
                "commentary": result.get("commentary", "") if result else ""
            },
            "key_signals": result.get("key_signals", "") if result else ""
        }
        print(f"  Direction: {mom_comparisons[topic]['sentiment_shift']['direction']}")

        time.sleep(2)

else:
    print("\nOnly one month found. MoM comparison requires 2+ months.")
    mom_comparisons = {}

# ============================================================
# CELL 6: BUILD DATA PAYLOAD
# ============================================================
print("\n" + "=" * 60)
print("CELL 6: BUILD DATA PAYLOAD")
print("=" * 60)

total_unique = len(all_unique)
total_pos = overall_sent.get("POSITIVE", 0)
total_neg = overall_sent.get("NEGATIVE", 0)
total_neu = overall_sent.get("NEUTRAL", 0)

dashboard_data = {
    "metadata": {
        "current_month": CURRENT_MONTH,
        "prior_month": PRIOR_MONTH or CURRENT_MONTH,
        "total_comments": total_unique
    },
    "topic_narratives": topic_narratives,
    "mom_comparisons": mom_comparisons
}

print(f"Unique comments: {total_unique}")
print(f"Sentiment: +{total_pos} ~{total_neu} -{total_neg}")
print(f"Topics: {len(topic_narratives)}")
print(f"MoM comparisons: {len(mom_comparisons)}")

# ============================================================
# CELL 7: GENERATE HTML OUTPUT
# ============================================================
print("\n" + "=" * 60)
print("CELL 7: GENERATE HTML OUTPUT")
print("=" * 60)

with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
    template = f.read()

data_json = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
html = template.replace("/*DATA*/", data_json)

with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to: {os.path.abspath(OUTPUT_HTML)}")

# ============================================================
# CELL 8: SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CELL 8: SUMMARY")
print("=" * 60)

print(f"Source: {DATA_FILE}")
print(f"Columns: {MONTH_COL} | {COMMENT_COL} | {TOPIC_COL} | {SUBTOPIC_COL} | {SENTIMENT_COL}")
print(f"Period: {PRIOR_MONTH or 'N/A'} \u2192 {CURRENT_MONTH}")
print(f"Rows loaded: {len(df)} | Unique comments: {total_unique}")
print(f"Sentiment: +{total_pos} / ~{total_neu} / -{total_neg}")
print(f"Topics: {len(topic_narratives)}")
print(f"MoM comparisons: {len(mom_comparisons)}")
print(f"Output: {OUTPUT_HTML}")
print("DONE!")
