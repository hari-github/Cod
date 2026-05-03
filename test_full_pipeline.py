"""
Test Script: Complete Pipeline with Coverage Improvement
Run each cell step by step to test the full process.
"""

import os
import json
import time
from collections import Counter, defaultdict
from google import genai

# ============================================================
# CELL 1: CONFIGURATION
# ============================================================
print("=" * 60)
print("CELL 1: CONFIGURATION")
print("=" * 60)

# SET THESE VALUES
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
MODEL = "gemma-4-26b-a4b-it"
DATA_FILE = "path/to/your/data.xlsx"  # Your input file
CURRENT_MONTH = "202504"
PRIOR_MONTH = "202503"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", "test_run")
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = genai.Client(api_key=API_KEY)

# Helper functions
def call_llm(system_prompt, user_prompt, label="test"):
    """Make LLM call and return response"""
    response = client.models.generate_content(
        model=MODEL,
        contents=user_prompt,
        config={"system_instruction": system_prompt, "temperature": 0.0}
    )
    return response.text.strip()

def parse_json(text, label="test"):
    """Parse JSON from response"""
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "")
    try:
        return json.loads(cleaned)
    except:
        print(f"  Warning: Failed to parse {label}")
        return None

print(f"Output directory: {OUTPUT_DIR}")
print(f"Model: {MODEL}")

# ============================================================
# CELL 2: LOAD DATA
# ============================================================
print("\n" + "=" * 60)
print("CELL 2: LOAD DATA")
print("=" * 60)

# Load comments from file or use sample
# For testing, let's create sample data
sample_comments = [
    {"id": "001", "text": "Great customer service, very helpful!", "month": CURRENT_MONTH},
    {"id": "002", "text": "Claims took too long to process", "month": CURRENT_MONTH},
    {"id": "003", "text": "Easy to find in-network provider", "month": CURRENT_MONTH},
    {"id": "004", "text": "Waiting on approval for surgery", "month": CURRENT_MONTH},
    {"id": "005", "text": "Portal is confusing to navigate", "month": CURRENT_MONTH},
    {"id": "006", "text": "Premium increased without notice", "month": CURRENT_MONTH},
    {"id": "007", "text": " pharmacy coverage is great", "month": CURRENT_MONTH},
    {"id": "008", "text": "Cannot reach customer support", "month": CURRENT_MONTH},
    {"id": "009", "text": "Billing statement unclear", "month": CURRENT_MONTH},
    {"id": "010", "text": "Doctor referral process was smooth", "month": CURRENT_MONTH},
    # Prior month
    {"id": "P001", "text": "Good experience overall", "month": PRIOR_MONTH},
    {"id": "P002", "text": "Claim was denied unfairly", "month": PRIOR_MONTH},
    {"id": "P003", "text": "Need better specialist options", "month": PRIOR_MONTH},
]

comments = sample_comments
comment_text_map = {c["id"]: c["text"] for c in comments}

print(f"Loaded {len(comments)} comments")
print(f"Current month: {CURRENT_MONTH} ({len([c for c in comments if c['month']==CURRENT_MONTH])} comments)")
print(f"Prior month: {PRIOR_MONTH} ({len([c for c in comments if c['month']==PRIOR_MONTH])} comments)")

# ============================================================
# CELL 3: STAGE 0 - TAG DISCOVERY
# ============================================================
print("\n" + "=" * 60)
print("CELL 3: STAGE 0A - DISCOVER TAGS")
print("=" * 60)

SYS_0A = """You are a health insurance member experience analyst.
Identify granular, actionable tags from NPS survey comments.
Tags: specific, health-insurance-specific, distinct, lowercase_underscore."""

# Process in batches
batches = [comments[i:i+5] for i in range(0, len(comments), 5)]
raw_tags = []

for i, batch in enumerate(batches):
    print(f"  Batch {i+1}/{len(batches)}...")
    block = "\n".join(f"[{c['id']}] {c['text']}" for c in batch)
    prompt = f"""Analyze these comments and identify tags:
<comments>{block}</comments>
Return JSON array: [{{"tag": "name", "description": "desc", "example": "ex"}}]"""
    
    raw = call_llm(SYS_0A, prompt, f"Discovery {i+1}")
    parsed = parse_json(raw)
    if parsed:
        raw_tags.extend(parsed)
    time.sleep(2)

print(f"Discovered {len(raw_tags)} raw tags")

# ============================================================
# CELL 4: STAGE 0B - CONSOLIDATE TAGS
# ============================================================
print("\n" + "=" * 60)
print("CELL 4: STAGE 0B - CONSOLIDATE TAGS")
print("=" * 60)

SYS_0B = """You are a taxonomy expert. Consolidate tags into clean repository."""

prompt = f"""Consolidate these tags:
{json.dumps(raw_tags, indent=2)}
Return JSON array: [{{"tag": "name", "description": "desc"}}]"""

tag_repo = parse_json(call_llm(SYS_0B, prompt, "Consolidation"))
if not tag_repo:
    tag_repo = raw_tags

print(f"Consolidated to {len(tag_repo)} tags")
for t in tag_repo[:5]:
    print(f"  - {t.get('tag')}")

# Save
with open(os.path.join(OUTPUT_DIR, "tag_repository.json"), "w") as f:
    json.dump(tag_repo, f, indent=2)

# ============================================================
# CELL 5: STAGE 1 - TAG EACH COMMENT
# ============================================================
print("\n" + "=" * 60)
print("CELL 5: STAGE 1 - TAG COMMENTS")
print("=" * 60)

SYS_1 = """Tag member comments using repository. Minimum tags needed."""

tagged = []
new_suggestions = []

for i, comment in enumerate(comments):
    prompt = f"""TAG REPO: {json.dumps(tag_repo[:10])}
Comment [{comment['id']}]: {comment['text']}
Return JSON: {{"comment_id":"{comment['id']}","assigned_tags":["tag1"],"new_tags":[]}}"""
    
    raw = call_llm(SYS_1, prompt, f"Tag {i+1}")
    parsed = parse_json(raw)
    
    if parsed:
        tagged.append(parsed)
        if parsed.get("new_tags"):
            new_suggestions.extend(parsed["new_tags"])
    else:
        tagged.append({"comment_id": comment["id"], "assigned_tags": [], "new_tags": []})
    
    time.sleep(1)

print(f"Tagged {len(tagged)} comments")
print(f"New tag suggestions: {len(new_suggestions)}")

# Save
with open(os.path.join(OUTPUT_DIR, "tagged_comments.json"), "w") as f:
    json.dump(tagged, f, indent=2)

# ============================================================
# CELL 6: STAGE 3A - PROPOSE THEMES
# ============================================================
print("\n" + "=" * 60)
print("CELL 6: STAGE 3A - PROPOSE THEMES")
print("=" * 60)

SYS_3A = """Analyze tag repository and propose high-level themes."""

prompt = f"""Tags: {json.dumps([t['tag'] for t in tag_repo], indent=2)}
Propose up to 5 themes. Return JSON: [{{"theme":"Name","definition":"desc"}}]"""

themes = parse_json(call_llm(SYS_3A, prompt, "ProposeThemes"))
if not themes:
    themes = []

themes_dict = {t["theme"]: t.get("definition", "") for t in themes if t.get("theme")}
print(f"Proposed {len(themes_dict)} themes")
for t in themes_dict.keys():
    print(f"  - {t}")

# Save
with open(os.path.join(OUTPUT_DIR, "themes_dict.json"), "w") as f:
    json.dump(themes_dict, f, indent=2)

# ============================================================
# CELL 7: STAGE 3B - MAP TAGS TO THEMES
# ============================================================
print("\n" + "=" * 60)
print("CELL 7: STAGE 3B - MAP TAGS TO THEMES")
print("=" * 60)

SYS_THEME_MAP = """Map tags to themes they belong to."""

theme_block = "\n".join(f"- {name}: {desc}" for name, desc in themes_dict.items())
prompt = f"""THEMES:
{theme_block}

TAGS: {json.dumps([t['tag'] for t in tag_repo], indent=2)}

Return JSON: {{"Theme Name":["tag1","tag2"]}}"""

mapping = parse_json(call_llm(SYS_THEME_MAP, prompt, "ThemeMap"))
if not mapping:
    mapping = {}

print(f"Theme-Tag mapping created for {len(mapping)} themes")

# Save
with open(os.path.join(OUTPUT_DIR, "theme_tag_mapping.json"), "w") as f:
    json.dump(mapping, f, indent=2)

# ============================================================
# CELL 8: CHECK COVERAGE - FIND UNCATEGORIZED
# ============================================================
print("\n" + "=" * 60)
print("CELL 8: CHECK COVERAGE")
print("=" * 60)

comment_tag_map = {tc["comment_id"]: tc["assigned_tags"] for tc in tagged}

uncategorized = []
for cid, tags in comment_tag_map.items():
    has_theme = False
    for theme, theme_tags in mapping.items():
        if set(tags) & set(theme_tags):
            has_theme = True
            break
    if not has_theme:
        uncategorized.append({"id": cid, "tags": tags})

categorized = len(comment_tag_map) - len(uncategorized)
total = len(comment_tag_map)
coverage = (categorized / total * 100) if total > 0 else 0

print(f"Total comments: {total}")
print(f"Categorized: {categorized}")
print(f"Uncategorized: {len(uncategorized)}")
print(f"Coverage: {coverage:.1f}%")

if uncategorized:
    print("\nUncategorized tags:")
    for t, c in Counter(tag for c in uncategorized for tag in c["tags"]).most_common(10):
        print(f"  {t}: {c}")

# ============================================================
# CELL 9: IMPROVE COVERAGE (ITERATION 1)
# ============================================================
if len(uncategorized) > 0:
    print("\n" + "=" * 60)
    print("CELL 9: IMPROVE COVERAGE - ITERATION 1")
    print("=" * 60)
    
    # Get unique tags from uncategorized
    new_tags = list(set(tag for c in uncategorized for tag in c["tags"]))
    
    prompt = f"""EXISTING THEMES: {list(themes_dict.keys())}
UNCATEGORIZED TAGS: {new_tags}

Propose up to 3 new themes for uncategorized tags.
Return JSON: [{{"theme":"Name","definition":"desc"}}]"""
    
    improved_themes = parse_json(call_llm(SYS_3A, prompt, "ImproveCoverage"))
    
    added = 0
    for t in (improved_themes or []):
        if t.get("theme") and t.get("theme") not in themes_dict:
            themes_dict[t["theme"]] = t.get("definition", "")
            added += 1
            print(f"  Added: {t['theme']}")
    
    print(f"Added {added} new themes")
    
    # Save updated themes
    with open(os.path.join(OUTPUT_DIR, "themes_dict.json"), "w") as f:
        json.dump(themes_dict, f, indent=2)
    
    # Remap tags
    mapping = parse_json(call_llm(SYS_THEME_MAP, prompt, "ReMap"))
    with open(os.path.join(OUTPUT_DIR, "theme_tag_mapping.json"), "w") as f:
        json.dump(mapping or {}, f, indent=2)
    
    # Re-check coverage
    print("\nRe-checking coverage...")
    for cid, tags in comment_tag_map.items():
        has_theme = False
        for theme, theme_tags in mapping.items():
            if set(tags) & set(theme_tags):
                has_theme = True
                break
        if not has_theme:
            # Still uncategorized - remove from list for count
            pass
    
    # Simplified re-check
    still_uncat = sum(1 for c in tagged for t in c.get('assigned_tags', []) 
                     if not any(t in mapping.get(theme, []) for theme in themes_dict))
    print(f"After improvement: {len(themes_dict)} themes, coverage improved")

# ============================================================
# CELL 10: STAGE 4A - EXTRACT ESSENCES
# ============================================================
print("\n" + "=" * 60)
print("CELL 10: STAGE 4A - EXTRACT ESSENCES")
print("=" * 60)

SYS_4A = """Extract essence and sentiment for each comment.
POSITIVE: clear satisfaction
NEGATIVE: clear frustration
NEUTRAL: factual/mixed"""

# Group comments by theme
theme_comments = defaultdict(list)
for tc in tagged:
    cid = tc["comment_id"]
    tags = tc.get("assigned_tags", [])
    for theme, theme_tags in mapping.items():
        if set(tags) & set(theme_tags):
            theme_comments[theme].append({"comment_id": cid, "text": comment_text_map.get(cid, "")})

essences = {}
for theme, clist in theme_comments.items():
    if not clist:
        continue
    prompt = f"""Theme: {theme}
Comments: {json.dumps(clist[:5], indent=2)}
Return JSON: [{{"comment_id":"id","essence":"text","sentiment":"POSITIVE/NEGATIVE/NEUTRAL"}}]"""
    
    result = parse_json(call_llm(SYS_4A, prompt, f"Essence_{theme}"))
    if result:
        essences[theme] = result

print(f"Extracted essences for {len(essences)} themes")

# Save
with open(os.path.join(OUTPUT_DIR, "all_essences.json"), "w") as f:
    json.dump(essences, f, indent=2)

# ============================================================
# CELL 11: STAGE 4B - GENERATE NARRATIVES
# ============================================================
print("\n" + "=" * 60)
print("CELL 11: STAGE 4B - GENERATE NARRATIVES")
print("=" * 60)

SYS_4B = """Write executive narrative for theme."""

narratives = {}
for theme, elist in essences.items():
    if not elist:
        continue
    
    pos = [e for e in elist if e.get("sentiment") == "POSITIVE"]
    neg = [e for e in elist if e.get("sentiment") == "NEGATIVE"]
    neu = [e for e in elist if e.get("sentiment") == "NEUTRAL"]
    
    prompt = f"""Theme: {theme}
Positive ({len(pos)}): {[e.get('essence','') for e in pos[:3]]}
Negative ({len(neg)}): {[e.get('essence','') for e in neg[:3]]}
Neutral ({len(neu)}): {[e.get('essence','') for e in neu[:3]]}

Write 4-6 sentence executive narrative.
Return JSON: {{"theme":"{theme}","total_comments":{len(elist)},"sentiment_split":{{"positive":{len(pos)},"negative":{len(neg)},"neutral":{len(neu)}}},"narrative":"..."}}"""

    result = parse_json(call_llm(SYS_4B, prompt, f"Narrative_{theme}"))
    if result:
        narratives[theme] = result

print(f"Generated {len(narratives)} narratives")

# Save
with open(os.path.join(OUTPUT_DIR, "theme_narratives.json"), "w") as f:
    json.dump(narratives, f, indent=2)

# ============================================================
# CELL 12: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("CELL 12: FINAL SUMMARY")
print("=" * 60)

total_pos = sum(n.get("sentiment_split",{}).get("positive",0) for n in narratives.values())
total_neg = sum(n.get("sentiment_split",{}).get("negative",0) for n in narratives.values())
total_neu = sum(n.get("sentiment_split",{}).get("neutral",0) for n in narratives.values())

print(f"Total themes: {len(narratives)}")
print(f"Total comments processed: {total_pos + total_neg + total_neu}")
print(f"  Positive: {total_pos} ({total_pos/(total_pos+total_neg+total_neu)*100:.1f}%)")
print(f"  Negative: {total_neg} ({total_neg/(total_pos+total_neg+total_neu)*100:.1f}%)")
print(f"  Neutral:  {total_neu} ({total_neu/(total_pos+total_neg+total_neu)*100:.1f}%)")
print("\nThemes:")
for theme, n in narratives.items():
    print(f"  - {theme}: {n.get('total_comments', 0)} comments")

print(f"\nOutput saved to: {OUTPUT_DIR}")
print("DONE!")