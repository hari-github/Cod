"""
Test Script: Improve Categorization Coverage
Run each cell step by step to test the improvement process.
"""

import os
import json
import pandas as pd
from google import genai

# ============================================================
# CELL 1: Setup - Configure API and Load Run Data
# ============================================================
print("=" * 60)
print("CELL 1: SETUP")
print("=" * 60)

# Configuration
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your key
MODEL = "gemma-4-26b-a4b-it"
RUN_ID = "e9e72aa7_0501_1802"  # Use the run you want to test

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output", RUN_ID)
print(f"Output directory: {OUTPUT_DIR}")

# Load existing data
tagged_comments = json.load(open(os.path.join(OUTPUT_DIR, "tagged_comments.json")))
themes_dict = json.load(open(os.path.join(OUTPUT_DIR, "themes_dict.json")))
mapping = json.load(open(os.path.join(OUTPUT_DIR, "theme_tag_mapping.json")))
comment_text_map = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
print(f"Loaded {len(tagged_comments)} tagged comments")
print(f"Loaded {len(themes_dict)} themes")
print(f"Loaded {len(mapping)} theme-tag mappings")

# ============================================================
# CELL 2: Identify Uncategorized Comments
# ============================================================
print("\n" + "=" * 60)
print("CELL 2: IDENTIFY UNCATEGORIZED COMMENTS")
print("=" * 60)

# Get all comment IDs and their tags
comment_tag_map = {tc["comment_id"]: tc["assigned_tags"] for tc in tagged_comments}
print(f"Total tagged comments: {len(comment_tag_map)}")

# Find comments not assigned to any theme
uncategorized = []
for cid, tags in comment_tag_map.items():
    has_theme = False
    for theme, theme_tags in mapping.items():
        if set(tags) & set(theme_tags):
            has_theme = True
            break
    if not has_theme:
        uncategorized.append({
            "id": cid,
            "tags": tags
        })

print(f"Uncategorized comments: {len(uncategorized)}")

# ============================================================
# CELL 3: Analyze Uncategorized Tags
# ============================================================
print("\n" + "=" * 60)
print("CELL 3: ANALYZE UNCATEGORIZED TAGS")
print("=" * 60)

# Get unique tags from uncategorized comments
new_tags = list(set(tag for c in uncategorized for tag in c["tags"]))
print(f"Unique tags from uncategorized: {len(new_tags)}")
print(f"Tags: {new_tags}")

# Group by frequency
from collections import Counter
tag_counter = Counter(tag for c in uncategorized for tag in c["tags"])
print("\nTop 10 most common uncategorized tags:")
for tag, count in tag_counter.most_common(10):
    print(f"  {tag}: {count}")

# ============================================================
# CELL 4: Create LLM Prompt for New Themes
# ============================================================
print("\n" + "=" * 60)
print("CELL 4: CREATE PROMPT FOR NEW THEMES")
print("=" * 60)

existing_themes = list(themes_dict.keys())
print(f"Existing themes: {existing_themes}")

# Build prompt
prompt = f"""You are a health insurance member experience analyst.
Based on these tags from uncategorized comments, propose new themes.

EXISTING THEMES: {existing_themes}

UNCATEGORIZED TAGS (with frequency):
{json.dumps(dict(tag_counter.most_common(15)), indent=2)}

TASK:
1. Review existing themes to avoid duplicates
2. Propose up to 3 NEW themes if the uncategorized tags represent distinct topics
3. Return ONLY a JSON array: [{{"theme": "Name", "definition": "Brief definition"}}]
4. If no new themes needed, return empty array []

Return only valid JSON."""

print("Prompt created:")
print(prompt[:500] + "...")

# ============================================================
# CELL 5: Call LLM to Get New Theme Suggestions
# ============================================================
print("\n" + "=" * 60)
print("CELL 5: CALL LLM FOR NEW THEMES")
print("=" * 60)

# Initialize client
client = genai.Client(api_key=API_KEY)

# Make the call
response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={
        "system_instruction": "You are a health insurance member experience analyst. Return only valid JSON.",
        "temperature": 0.0,
    }
)

print(f"Response received:")
print(response.text)

# Parse response
import re
cleaned = re.sub(r"```json\s*", "", response.text)
cleaned = re.sub(r"```\s*", "", cleaned).strip()

try:
    new_themes = json.loads(cleaned)
    print(f"\nParsed {len(new_themes)} new theme(s)")
    for t in new_themes:
        print(f"  - {t.get('theme')}: {t.get('definition', '')[:50]}...")
except json.JSONDecodeError as e:
    print(f"Failed to parse JSON: {e}")
    new_themes = []

# ============================================================
# CELL 6: Add New Themes to Existing Themes
# ============================================================
print("\n" + "=" * 60)
print("CELL 6: ADD NEW THEMES")
print("=" * 60)

added_themes = []
for t in new_themes:
    theme_name = t.get("theme", "").strip()
    if theme_name and theme_name not in themes_dict:
        themes_dict[theme_name] = t.get("definition", "")
        added_themes.append(theme_name)
        print(f"Added: {theme_name}")
    else:
        print(f"Skipped (duplicate or empty): {theme_name}")

print(f"\nTotal new themes added: {len(added_themes)}")
print(f"Total themes now: {len(themes_dict)}")

# Save updated themes
if added_themes:
    output_path = os.path.join(OUTPUT_DIR, "themes_dict.json")
    with open(output_path, "w") as f:
        json.dump(themes_dict, f, indent=2)
    print(f"\nSaved updated themes to: {output_path}")

# ============================================================
# CELL 7: Show Summary
# ============================================================
print("\n" + "=" * 60)
print("CELL 7: SUMMARY")
print("=" * 60)

print(f"Original themes: {len(existing_themes)}")
print(f"New themes added: {len(added_themes)}")
print(f"Total themes: {len(themes_dict)}")
print(f"Uncategorized comments before: {len(uncategorized)}")
print(f"Remaining uncategorized: {len(uncategorized)} (will need re-tagging)")
print("\nNew theme list:")
for t in themes_dict.keys():
    marker = "✓" if t in added_themes else " "
    print(f"  [{marker}] {t}")