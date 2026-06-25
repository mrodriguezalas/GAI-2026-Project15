import faiss
import numpy as np
from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("API key is missing")  

client = OpenAI()

# Load FAISS index + file metadata safely
# Note: Point these paths to where you saved your files in the embedding script!
INDEX_PATH = "src/pdf_chunks.index"
METADATA_PATH = "src/pdf_metadata.npy"

if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
    print("❌ Error: Missing index or metadata files. Run create_embedding.py first.")
    exit()

index = faiss.read_index(INDEX_PATH)
metadata_records = np.load(METADATA_PATH, allow_pickle=True)

# --- Embed a query ---
def embed_query(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    vec = np.array(response.data[0].embedding).astype("float32")
    return vec

# --- Search function ---
def search(query, k=5):
    qvec = embed_query(query).reshape(1, -1)
    distances, indices = index.search(qvec, k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if idx == -1: 
            continue
            
        meta = metadata_records[idx]
        
        results.append({
            "rank": rank,
            "pdf_name": meta["pdf_name"],
            "page": meta["page"],
            "lines": meta["lines"],
            "text": meta["text"],  # <--- Pull the raw text here
            "distance_score": float(dist)
        })

    return results

# --- Example Pipeline Executing Your Output Format ---
'''if __name__ == "__main__":
    user_query = "What problem does the Transformer solve compared to RNNs?"
    top_hits = search(user_query, k=5)

    # Print out just the top match using your requested format rules:
    if top_hits:
        best_match = top_hits[0]
        
        print(f"[{best_match['pdf_name']}]")
        print(f"{best_match['text']} [Page:{best_match['page']}] [Lines:{best_match['lines']}] \n")'''