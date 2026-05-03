import json
import time
import os
import numpy as np
from google import genai
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
MODEL_TEXT = "gemini-2.0-flash" 
MODEL_EMBED = "text-embedding-004"

client = genai.Client(api_key=API_KEY)

# ============================================================
# 1. MOCK DATA (Simulating large Stage 0A output)
# ============================================================
def get_raw_tags():
    return [
        {"tag": "billing_issue", "description": "General problems with billing statements"},
        {"tag": "statement_confusion", "description": "Member cannot understand the layout of the bill"},
        {"tag": "incorrect_bill", "description": "Member claims the amount charged is wrong"},
        {"tag": "slow_claims", "description": "Claims processing is taking too long"},
        {"tag": "claim_delay", "description": "Significant wait time for claim reimbursement"},
        {"tag": "processing_lag", "description": "Back-end processing of claims is slow"},
        {"tag": "app_crash", "description": "The mobile application closes unexpectedly"},
        {"tag": "mobile_bug", "description": "Errors found within the digital app"},
        {"tag": "portal_error", "description": "Web portal is returning technical errors"},
        {"tag": "friendly_agent", "description": "The customer service rep was very nice"},
        {"tag": "polite_support", "description": "Support staff was courteous and helpful"},
        {"tag": "nice_representative", "description": "The person I spoke to was very pleasant"},
    ]

# ============================================================
# 2. EMBEDDING GENERATION
# ============================================================
def get_embeddings(tag_list):
    print(f"Generating embeddings for {len(tag_list)} tags...")
    embeddings = []
    for t in tag_list:
        text = f"{t['tag']}: {t['description']}"
        try:
            response = client.models.embed_content(
                model=MODEL_EMBED,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"}
            )
            embeddings.append(response.embeddings[0].values)
        except Exception as e:
            print(f"  Error embedding {t['tag']}: {e}")
            embeddings.append([0.0] * 768) # Fallback vector
    return np.array(embeddings)

# ============================================================
# 3. SIMILARITY GROUPING (Simplified Clustering)
# ============================================================
def group_tags(tag_list, embeddings, threshold=0.75):
    """
    Groups tags based on cosine similarity.
    Tags with similarity > threshold are grouped together.
    """
    print(f"Grouping tags (threshold={threshold})...")
    clusters = []
    visited = set()

    for i in range(len(tag_list)):
        if i in visited:
            continue
        
        current_cluster = [i]
        visited.add(i)
        
        for j in range(i + 1, len(tag_list)):
            if j in visited:
                continue
            
            # Cosine Similarity calculation
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            if norm_i == 0 or norm_j == 0:
                sim = 0
            else:
                sim = np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j)
            
            if sim > threshold:
                current_cluster.append(j)
                visited.add(j)
        
        clusters.append([tag_list[idx] for idx in current_cluster])
    
    return clusters

# ============================================================
# 4. LLM CONSOLIDATION (Iterative)
# ============================================================
def consolidate_cluster(cluster_tags):
    if len(cluster_tags) == 1:
        return cluster_tags[0]
    
    print(f"  Consolidating cluster of {len(cluster_tags)} tags...")
    
    sys_prompt = """You are a taxonomy expert. 
    Merge highly similar tags into a single clean, descriptive tag.
    Use lowercase_underscore for the tag name.
    Preserve the core meaning while being concise."""
    
    user_prompt = f"""Consolidate these similar tags into ONE master tag:
    {json.dumps(cluster_tags, indent=2)}
    
    Return ONLY JSON: {{"tag": "master_tag_name", "description": "consolidated description"}}"""
    
    try:
        response = client.models.generate_content(
            model=MODEL_TEXT,
            contents=user_prompt,
            config={"system_instruction": sys_prompt, "response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"  Error in LLM consolidation: {e}")
        return cluster_tags[0] # Fallback

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your API_KEY in the script first.")
        return

    raw_tags = get_raw_tags()
    
    # 1. Get Embeddings
    embeddings = get_embeddings(raw_tags)
    
    # 2. Cluster
    # A threshold of 0.8 - 0.85 usually works well for semantic similarity
    clusters = group_tags(raw_tags, embeddings, threshold=0.82)
    print(f"Created {len(clusters)} clusters.")
    
    # 3. Iterative Consolidation
    consolidated_repo = []
    for i, cluster in enumerate(clusters):
        print(f"Processing Cluster {i+1}...")
        master_tag = consolidate_cluster(cluster)
        consolidated_repo.append(master_tag)
        time.sleep(1) # API Rate limit safety
        
    # 4. Final Result
    print("\n" + "="*50)
    print("FINAL CONSOLIDATED REPOSITORY")
    print("="*50)
    for t in consolidated_repo:
        print(f"- {t['tag']}: {t['description']}")

if __name__ == "__main__":
    main()
