import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

# Initialize the Gemini client
client = genai.Client()

# 1. Our mock database of survey comments
documents = [
    "I had to wait three whole weeks just to get my heart medication approved. Ridiculous.",
    "The customer service agent was nice, but they overcharged me for an out-of-network specialist visit.",
    "Why does my plan not cover physical therapy sessions? I have terrible back pain.",
    "The prior authorization process is completely broken and takes forever.",
]

# The user's actual search term
query = "prior auth delay"


# ==========================================
# STEP 1: KEYWORD SEARCH (Sparse Vector)
# ==========================================
# We use TF-IDF to find exact word matches in the documents
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
query_tfidf = vectorizer.transform([query])

# Calculate how well the exact keywords match (0.0 to 1.0)
keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()


# ==========================================
# STEP 2: SEMANTIC SEARCH (Dense Vector)
# ==========================================
# A helper function to call the Gemini API for embeddings
def get_embedding(text):
    response = client.models.embed_content(
        model="text-embedding-004", 
        contents=text
    )
    return response.embeddings[0].values

print("Generating Gemini embeddings...\n")
doc_embeddings = np.array([get_embedding(doc) for doc in documents])
query_embedding = np.array(get_embedding(query)).reshape(1, -1)

# Calculate how closely the concepts match mathematically
semantic_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()


# ==========================================
# STEP 3: HYBRID SEARCH COMBINATION
# ==========================================
# Normalize both scores to a 0-1 scale so they can be combined fairly
def normalize(scores):
    if np.max(scores) == 0: return scores
    return scores / np.max(scores)

norm_keyword = normalize(keyword_scores)
norm_semantic = normalize(semantic_scores)

# Alpha Weighting: Decide what matters more. 
# Here we say conceptual meaning is 70% important, exact keywords are 30%.
semantic_weight = 0.7
keyword_weight = 0.3

hybrid_scores = (norm_semantic * semantic_weight) + (norm_keyword * keyword_weight)


# ==========================================
# STEP 4: RANK AND RETURN RESULTS
# ==========================================
print(f"Results for query: '{query}'\n")

# Sort the array from highest hybrid score to lowest
ranked_indices = np.argsort(hybrid_scores)[::-1]

for idx in ranked_indices:
    print(f"Total Score: {hybrid_scores[idx]:.3f} | {documents[idx]}")
    print(f"  (Semantic Score: {norm_semantic[idx]:.2f}, Keyword Score: {norm_keyword[idx]:.2f})\n")
