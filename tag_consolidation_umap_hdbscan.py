import json
import time
import os
import numpy as np
import pandas as pd
from google import genai
from collections import defaultdict

# NOTE: You must install these extra libraries for this script:
# pip install umap-learn hdbscan

try:
    import umap
    import hdbscan
except ImportError:
    print("Error: 'umap-learn' or 'hdbscan' not found.")
    print("Please run: pip install umap-learn hdbscan")

# ============================================================
# CONFIGURATION
# ============================================================
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your API key
MODEL_TEXT = "gemini-2.0-flash" 
MODEL_EMBED = "text-embedding-004"

client = genai.Client(api_key=API_KEY)

# ============================================================
# 1. MOCK DATA (Expanded for HDBSCAN min_cluster_size=5)
# ============================================================
def get_raw_tags():
    return [
        # Billing Cluster
        {"tag": "billing_issue", "description": "General problems with billing statements"},
        {"tag": "statement_confusion", "description": "Member cannot understand the layout of the bill"},
        {"tag": "incorrect_bill", "description": "Member claims the amount charged is wrong"},
        {"tag": "payment_failed", "description": "Auto-pay or credit card payment did not go through"},
        {"tag": "billing_cycle_error", "description": "Bill arrived at the wrong time of the month"},
        {"tag": "double_charge", "description": "Member was billed twice for the same service"},
        
        # Claims Cluster
        {"tag": "slow_claims", "description": "Claims processing is taking too long"},
        {"tag": "claim_delay", "description": "Significant wait time for claim reimbursement"},
        {"tag": "processing_lag", "description": "Back-end processing of claims is slow"},
        {"tag": "denied_claim_appeal", "description": "Member wants to contest a claim rejection"},
        {"tag": "missing_claim_info", "description": "Claim is stuck because of missing provider details"},
        {"tag": "reimbursement_mismatch", "description": "The amount paid for the claim is less than expected"},
        
        # Digital Cluster
        {"tag": "app_crash", "description": "The mobile application closes unexpectedly"},
        {"tag": "mobile_bug", "description": "Errors found within the digital app"},
        {"tag": "portal_error", "description": "Web portal is returning technical errors"},
        {"tag": "login_failure", "description": "Member cannot log into the mobile app"},
        {"tag": "password_reset_issue", "description": "The forgot password link is not working"},
        {"tag": "app_performance", "description": "The mobile app is extremely slow and laggy"},
        
        # Support Cluster
        {"tag": "friendly_agent", "description": "The customer service rep was very nice"},
        {"tag": "polite_support", "description": "Support staff was courteous and helpful"},
        {"tag": "nice_representative", "description": "The person I spoke to was very pleasant"},
        {"tag": "helpful_operator", "description": "The agent answered all my questions clearly"},
        {"tag": "professional_staff", "description": "The support team acted very professionally"},
        {"tag": "competent_support", "description": "The agent knew exactly how to solve my issue"},
        
        # Network Cluster
        {"tag": "out_of_network_fees", "description": "Surprise costs from non-network doctors"},
        {"tag": "provider_not_found", "description": "Member cannot find a specialist in their area"},
        {"tag": "doctor_not_in_network", "description": "Member's preferred doctor is not covered"},
        {"tag": "narrow_network", "description": "Not enough local options for primary care"},
        {"tag": "referral_required", "description": "Member frustrated they need a referral for a specialist"},
        {"tag": "authorization_delay", "description": "Waiting too long for pre-approval for a procedure"},
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
            embeddings.append([0.0] * 768) 
    return np.array(embeddings)

# ============================================================
# 3. UMAP + HDBSCAN CLUSTERING
# ============================================================
def cluster_with_umap_hdbscan(tag_list, embeddings):
    print("Performing UMAP reduction and HDBSCAN clustering...")
    
    # Step 2: Reduce dimensions (using User's parameters)
    reducer = umap.UMAP(n_components=10, metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)

    # Step 3: Cluster (using User's parameters)
    # Note: min_cluster_size=5 means we need at least 5 similar tags to form a group
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    labels = clusterer.fit_predict(reduced)

    # Organise results
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(tag_list[i])
    
    return clusters

# ============================================================
# 4. LLM CONSOLIDATION
# ============================================================
def consolidate_cluster(cluster_tags):
    if len(cluster_tags) == 1: return cluster_tags[0]
    sys_prompt = "You are a taxonomy expert. Merge highly similar tags into a single clean, descriptive tag."
    user_prompt = f"Consolidate these similar tags into ONE master tag:\n{json.dumps(cluster_tags, indent=2)}\nReturn ONLY JSON: {{\"tag\": \"name\", \"description\": \"desc\"}}"
    try:
        response = client.models.generate_content(model=MODEL_TEXT, contents=user_prompt, config={"system_instruction": sys_prompt, "response_mime_type": "application/json"})
        return json.loads(response.text)
    except: return cluster_tags[0]

def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("Please set your API_KEY in the script first.")
        return
    
    raw_tags = get_raw_tags()
    embeddings = get_embeddings(raw_tags)
    
    # 3. Cluster
    clusters_map = cluster_with_umap_hdbscan(raw_tags, embeddings)
    
    print(f"\nCreated {len(clusters_map)} groups (Label -1 indicates noise/unclustered).")
    
    # 4. Iterative Consolidation
    consolidated_repo = []
    for label, cluster in clusters_map.items():
        if label == -1:
            print(f"Skipping {len(cluster)} unclustered tags (Noise)...")
            # Usually we keep noise tags as individual items
            consolidated_repo.extend(cluster)
            continue
            
        print(f"Processing Cluster {label} ({len(cluster)} tags)...")
        master_tag = consolidate_cluster(cluster)
        consolidated_repo.append(master_tag)
        time.sleep(1) 

    # 5. Final Result
    print("\n" + "="*50)
    print("FINAL REPOSITORY (UMAP + HDBSCAN)")
    print("="*50)
    for t in consolidated_repo: print(f"- {t['tag']}")

if __name__ == "__main__":
    main()
