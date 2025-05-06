from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
import uuid

client = QdrantClient(host="localhost", port=6333)

def setup_collection(collection_name: str, vector_dim: int):
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name = collection_name,
            vectors_config = VectorParams(size=vector_dim, distance=Distance.COSINE)
        )
        
COLLECTION_NAME = "user_embeddings"

def store_user_embedding(name, embedding, liveness_score):
    user_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(
            id=user_id,
            vector=embedding,
            payload={"name": name, "liveness_score": liveness_score}
        )]
    )
    return user_id
