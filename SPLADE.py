import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client import models

class SpladeVectorDB:
    def __init__(self, collection_name="health_comments"):
        self.collection_name = collection_name
        
        # Initialize an in-memory database for testing in Databricks.
        # In production, replace this with a Qdrant Cloud URL and API key.
        self.client = QdrantClient(":memory:") 
        
        # Databricks tip: In production, load this model via DBFS or 
        # Unity Catalog Volumes to avoid re-downloading on cluster restarts.
        print("Loading SPLADE model...")
        self.tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3-distilbert")
        self.model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3-distilbert")
        self.model.eval()
        
        # Create a database collection specifically configured for SPARSE vectors
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={}, # Explicitly telling the DB we are NOT using dense vectors
                sparse_vectors_config={
                    "splade_vector": models.SparseVectorParams()
                }
            )

    def _compute_sparse_vector(self, text: str):
        """Converts text into database-ready sparse indices and weights."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**inputs)
            
        logits = output.logits
        relu_log = torch.log(1 + torch.relu(logits))
        sparse_vec, _ = torch.max(relu_log, dim=1)
        sparse_vec = sparse_vec.squeeze()
        
        # Extract only the non-zero tokens for database storage
        non_zero_indices = sparse_vec.nonzero().squeeze()
        if non_zero_indices.dim() == 0: 
            non_zero_indices = non_zero_indices.unsqueeze(0)
            
        weights = sparse_vec[non_zero_indices]
        
        # Format required by Vector Databases: Separate lists for active indices and their scores
        return {
            "indices": non_zero_indices.tolist(),
            "values": weights.tolist()
        }

    def index_documents(self, documents: list[str]):
        """Embeds documents and upserts them into the external Vector DB."""
        print(f"Indexing {len(documents)} documents to Vector DB...")
        
        points = []
        for idx, text in enumerate(documents):
            sparse_data = self._compute_sparse_vector(text)
            
            # Create a database point. The actual text is stored in the "payload" (metadata)
            point = models.PointStruct(
                id=idx + 1, # DB requires integer or UUID
                vector={
                    "splade_vector": models.SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"]
                    )
                },
                payload={"original_text": text}
            )
            points.append(point)
            
        # Push to the database instance
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print("Upsert complete.")

    def search(self, query: str, top_k: int = 3):
        """Queries the Vector DB for the closest sparse matches."""
        query_data = self._compute_sparse_vector(query)
        
        # The database calculates the dot product internally based on the sparse indices
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=models.NamedSparseVector(
                name="splade_vector",
                vector=models.SparseVector(
                    indices=query_data["indices"],
                    values=query_data["values"]
                )
            ),
            limit=top_k
        )
        
        # Parse the database response and retrieve the text from the payload
        return [(hit.score, hit.payload["original_text"]) for hit in results]
