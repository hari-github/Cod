"""
NPS Dashboard Generator
Takes a DataFrame with Survey Month, Comments, Topics, Subtopics columns,
calls Gemini API to generate summaries, and outputs an HTML dashboard.
"""

import os
import re
import json
import time
import pandas as pd
from google import genai
from collections import defaultdict

# ============================================================
# CONFIGURATION - Set these values
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"
MODEL = "gemma-4-26b-a4b-it"
DATA_FILE = "data.xlsx"  # Path to your input file
SHEET_NAME = "Sheet1"    # If using Excel

# Column names in your DataFrame
MONTH_COL = "Survey Month"
COMMENT_COL = "Comments"
TOPIC_COL = "Topics"
SUBTOPIC_COL = "Subtopics"

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
for col in [MONTH_COL, COMMENT_COL, TOPIC_COL, SUBTOPIC_COL]:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in data. Available: {list(df.columns)}")

# Detect current and prior months
months = sorted(df[MONTH_COL].unique())
CURRENT_MONTH = months[-1]
PRIOR_MONTH = months[-2] if len(months) > 1 else None
print(f"Current month: {CURRENT_MONTH}")
print(f"Prior month: {PRIOR_MONTH}")

# Build structured data
comments = []
for _, row in df.iterrows():
    comments.append({
        "id": str(row.name),
        "month": str(row[MONTH_COL]),
        "text": str(row[COMMENT_COL]),
        "topic": str(row[TOPIC_COL]),
        "subtopic": str(row[SUBTOPIC_COL])
    })

# Group by topic
topic_groups = defaultdict(list)
for c in comments:
    topic_groups[c["topic"]].append(c)

print(f"Topics found: {list(topic_groups.keys())}")

# ============================================================
# CELL 2: GENERATE TOPIC NARRATIVES
# ============================================================
print("\n" + "=" * 60)
print("CELL 2: GENERATE TOPIC NARRATIVES")
print("=" * 60)

SYS_TOPIC = """You are a health insurance member experience analyst.
Analyze NPS feedback comments for a given topic and produce structured output.
Be specific, evidence-driven, and concise."""

topic_narratives = {}

for topic, topic_comments in topic_groups.items():
    print(f"\nProcessing topic: {topic} ({len(topic_comments)} comments)")

    # Get unique subtopics for this topic
    subtopics = sorted(set(c["subtopic"] for c in topic_comments if c["subtopic"]))

    # Build comment block with IDs
    comment_block = "\n".join(
        f"[{c['id']}] (Month: {c['month']}, Subtopic: {c['subtopic']}) {c['text']}"
        for c in topic_comments
    )

    prompt = f"""Topic: {topic}
Subtopics identified: {subtopics if subtopics else "None specified"}

Comments:
{comment_block}

Return JSON with this exact structure:
{{
  "sentiment_split": {{"positive": count, "negative": count, "neutral": count}},
  "narrative": "4-6 sentence executive summary of member feedback for this topic. Include specific patterns, key drivers, and actionable insights.",
  "representative_quotes": {{
    "negative": ["quote 1", "quote 2"],
    "positive": ["quote 1", "quote 2"]
  }}
}}

Rules:
- Classify each comment's sentiment accurately
- Pick quotes that best represent the sentiment direction
- Narrative must be specific to the data, not generic
- Total of positive+negative+neutral must equal {len(topic_comments)}"""

    raw = call_llm(SYS_TOPIC, prompt, f"Topic: {topic}")
    result = parse_json(raw, f"Topic: {topic}")

    if result and result.get("sentiment_split"):
        topic_narratives[topic] = {
            "topic": topic,
            "total_comments": len(topic_comments),
            "sentiment_split": result["sentiment_split"],
            "subtopics": subtopics,
            "narrative": result.get("narrative", ""),
            "representative_quotes": result.get("representative_quotes", {"negative": [], "positive": []})
        }
        s = result["sentiment_split"]
        print(f"  Pos:{s.get('positive',0)} Neu:{s.get('neutral',0)} Neg:{s.get('negative',0)}")
    else:
        topic_narratives[topic] = {
            "topic": topic, "total_comments": len(topic_comments),
            "sentiment_split": {"positive": 0, "negative": 0, "neutral": len(topic_comments)},
            "subtopics": subtopics, "narrative": "", "representative_quotes": {"negative": [], "positive": []}
        }
        print(f"  Failed to parse, using defaults")

    time.sleep(2)

print(f"\nGenerated {len(topic_narratives)} topic narratives")

# ============================================================
# CELL 3: GENERATE MOM COMPARISONS
# ============================================================
print("\n" + "=" * 60)
print("CELL 3: GENERATE MONTH-OVER-MONTH COMPARISONS")
print("=" * 60)

SYS_MOM = """You are a health insurance member experience analyst.
Compare sentiment between two months and identify signals and shifts.
Be concise, data-driven, and specific."""

mom_comparisons = {}

if PRIOR_MONTH:
    for topic in topic_narratives.keys():
        # Split comments by month
        current_tc = [c for c in topic_groups[topic] if c["month"] == CURRENT_MONTH]
        prior_tc = [c for c in topic_groups[topic] if c["month"] == PRIOR_MONTH]

        if not current_tc or not prior_tc:
            print(f"  Skipping {topic} (no data in both months)")
            continue

        print(f"\nComparing {topic}: Current={len(current_tc)}, Prior={len(prior_tc)}")

        # Build prompt with samples from both months
        curr_block = "\n".join(f"[{c['id']}] {c['text']}" for c in current_tc[:10])
        prior_block = "\n".join(f"[{c['id']}] {c['text']}" for c in prior_tc[:10])

        prompt = f"""Topic: {topic}

Current Month ({CURRENT_MONTH}) - {len(current_tc)} comments:
{curr_block}

Prior Month ({PRIOR_MONTH}) - {len(prior_tc)} comments:
{prior_block}

Return JSON with this exact structure:
{{
  "sentiment_prior": {{"positive": count, "negative": count, "neutral": count}},
  "sentiment_current": {{"positive": count, "negative": count, "neutral": count}},
  "direction": "IMPROVED or WORSENED or STABLE",
  "commentary": "2-3 sentence explanation of the sentiment shift and what drove it",
  "key_signals": "1-2 sentence key takeaway for leadership"
}}

Rules:
- IMPROVED if negative sentiment decreased or positive increased significantly
- WORSENED if negative sentiment increased or positive decreased significantly
- STABLE if no meaningful change
- Commentary must reference specific patterns from the data
- sentiment_prior total must equal {len(prior_tc)}
- sentiment_current total must equal {len(current_tc)}"""

        raw = call_llm(SYS_MOM, prompt, f"MoM: {topic}")
        result = parse_json(raw, f"MoM: {topic}")

        # Get prior narrative
        prior_prompt = f"""Topic: {topic}
Summarize this prior month feedback in 3-4 sentences:

Comments ({PRIOR_MONTH}):
{prior_block}

Return JSON: {{"narrative": "3-4 sentence summary"}}"""

        prior_raw = call_llm(SYS_TOPIC, prior_prompt, f"Prior: {topic}")
        prior_result = parse_json(prior_raw, f"Prior: {topic}")

        mom_comparisons[topic] = {
            "topic": topic,
            "prior_narrative": prior_result.get("narrative", "") if prior_result else "",
            "sentiment_shift": {
                "prior": result.get("sentiment_prior", {"positive": 0, "negative": 0, "neutral": len(prior_tc)}) if result else {"positive": 0, "negative": 0, "neutral": len(prior_tc)},
                "current": result.get("sentiment_current", {"positive": 0, "negative": 0, "neutral": len(current_tc)}) if result else {"positive": 0, "negative": 0, "neutral": len(current_tc)},
                "direction": result.get("direction", "STABLE") if result else "STABLE",
                "commentary": result.get("commentary", "") if result else ""
            },
            "key_signals": result.get("key_signals", "") if result else ""
        }
        print(f"  Direction: {mom_comparisons[topic]['sentiment_shift']['direction']}")

        time.sleep(2)

else:
    print("Only one month found. MoM comparison requires at least 2 months of data.")
    mom_comparisons = {}

# ============================================================
# CELL 4: BUILD DATA PAYLOAD
# ============================================================
print("\n" + "=" * 60)
print("CELL 4: BUILD DATA PAYLOAD")
print("=" * 60)

total_all = sum(t["total_comments"] for t in topic_narratives.values())

# Get sentiment counts from the LLM-generated splits
total_pos = sum(t["sentiment_split"].get("positive", 0) for t in topic_narratives.values())
total_neg = sum(t["sentiment_split"].get("negative", 0) for t in topic_narratives.values())
total_neu = sum(t["sentiment_split"].get("neutral", 0) for t in topic_narratives.values())

dashboard_data = {
    "metadata": {
        "current_month": CURRENT_MONTH,
        "prior_month": PRIOR_MONTH or CURRENT_MONTH,
        "total_comments": total_all
    },
    "topic_narratives": topic_narratives,
    "mom_comparisons": mom_comparisons
}

print(f"Total comments: {total_all}")
print(f"Topics: {len(topic_narratives)}")
print(f"MoM comparisons: {len(mom_comparisons)}")

# ============================================================
# CELL 5: GENERATE HTML OUTPUT
# ============================================================
print("\n" + "=" * 60)
print("CELL 5: GENERATE HTML OUTPUT")
print("=" * 60)

# Load template
with open(TEMPLATE_HTML, "r", encoding="utf-8") as f:
    template = f.read()

# Inject data
data_json = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
html = template.replace("/*DATA*/", data_json)

# Save
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Dashboard saved to: {os.path.abspath(OUTPUT_HTML)}")

# ============================================================
# CELL 6: SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CELL 6: SUMMARY")
print("=" * 60)

print(f"Source data: {DATA_FILE}")
print(f"Column mapping:")
print(f"  {MONTH_COL} -> Survey Month")
print(f"  {COMMENT_COL} -> Comments")
print(f"  {TOPIC_COL} -> Topics")
print(f"  {SUBTOPIC_COL} -> Subtopics")
print(f"Period: {PRIOR_MONTH or 'N/A'} -> {CURRENT_MONTH}")
print(f"Total comments: {total_all}")
print(f"Sentiment: +{total_pos} / ~{total_neu} / -{total_neg}")
print(f"Topics analyzed: {len(topic_narratives)}")
print(f"MoM comparisons: {len(mom_comparisons)}")
print(f"Output: {OUTPUT_HTML}")
print("DONE!")
